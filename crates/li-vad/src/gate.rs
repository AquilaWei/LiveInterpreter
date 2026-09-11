//! The decision layer: speech probabilities in, gate state and events out.
//!
//! Kept apart from the model on purpose. Everything that has ever gone wrong
//! with a VAD in this project has been timing and state, not arithmetic, and
//! this half can be tested exhaustively against synthetic probabilities while
//! the ONNX half only has to be checked once against real audio.

use std::time::Duration;

use crate::VadEvent;

#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Probability at or above which a window counts as speech.
    pub speech_threshold: f32,
    /// Probability below which a window counts as silence. Lower than
    /// `speech_threshold` on purpose: a single threshold makes the gate flap
    /// once per window on any audio that sits near it.
    pub silence_threshold: f32,
    /// Trailing silence before [`VadEvent::SpeechEnd`]. Speakers stop for the
    /// closure of a /t/ or /k/; ending the run there would cut words in half.
    pub min_silence: Duration,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence_threshold: 0.35,
            min_silence: Duration::from_millis(200),
        }
    }
}

/// Speech / silence state machine over a stream of scored windows.
#[derive(Debug)]
pub struct Gate {
    cfg: GateConfig,
    speaking: bool,
    /// Time since the last window that was *not* silence.
    silence: Duration,
    last_prob: f32,
}

impl Gate {
    pub fn new(cfg: GateConfig) -> Self {
        Self {
            cfg,
            speaking: false,
            silence: Duration::ZERO,
            last_prob: 0.0,
        }
    }

    /// Score one window. `dur` is how much audio it covered.
    pub fn push(&mut self, prob: f32, dur: Duration) -> Option<VadEvent> {
        self.last_prob = prob;

        // Silence is measured against the *lower* threshold and independently
        // of the state machine, so `silence()` keeps counting through the
        // `min_silence` hold. `li-stream` compares it to `pause_flush`, which
        // is longer, and would otherwise see the timer restart under it.
        if prob >= self.cfg.silence_threshold {
            self.silence = Duration::ZERO;
        } else {
            self.silence += dur;
        }

        if !self.speaking {
            // Enter on the first speech window, with no debounce: the onset of
            // a sentence is exactly the audio the recogniser must not miss.
            if prob >= self.cfg.speech_threshold {
                self.speaking = true;
                return Some(VadEvent::SpeechStart);
            }
        } else if self.silence >= self.cfg.min_silence {
            self.speaking = false;
            return Some(VadEvent::SpeechEnd);
        }
        None
    }

    pub fn is_speech(&self) -> bool {
        self.speaking
    }

    pub fn silence(&self) -> Duration {
        self.silence
    }

    /// The most recent window's probability. For the UI level meter and for
    /// working out why a gate did what it did.
    pub fn probability(&self) -> f32 {
        self.last_prob
    }

    pub fn reset(&mut self) {
        self.speaking = false;
        self.silence = Duration::ZERO;
        self.last_prob = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: Duration = Duration::from_millis(32);

    fn feed(gate: &mut Gate, probs: &[f32]) -> Vec<VadEvent> {
        probs.iter().filter_map(|&p| gate.push(p, W)).collect()
    }

    #[test]
    fn speech_starts_on_the_first_speech_window() {
        let mut g = Gate::new(GateConfig::default());
        assert_eq!(feed(&mut g, &[0.0, 0.1, 0.9]), vec![VadEvent::SpeechStart]);
        assert!(g.is_speech());
        assert_eq!(g.silence(), Duration::ZERO);
    }

    #[test]
    fn a_stop_consonant_does_not_end_the_run() {
        let mut g = Gate::new(GateConfig::default());
        // 0.2s of hold at 32ms a window is 7 windows; five of silence is a
        // plosive closure, not the end of a sentence.
        assert_eq!(feed(&mut g, &[0.9]), vec![VadEvent::SpeechStart]);
        assert_eq!(feed(&mut g, &[0.0; 5]), vec![]);
        assert!(g.is_speech());
        assert_eq!(feed(&mut g, &[0.9, 0.9]), vec![]);
        assert!(g.is_speech());
    }

    #[test]
    fn speech_ends_after_sustained_silence() {
        let mut g = Gate::new(GateConfig::default());
        feed(&mut g, &[0.9]);
        assert_eq!(feed(&mut g, &[0.0; 7]), vec![VadEvent::SpeechEnd]);
        assert!(!g.is_speech());
    }

    #[test]
    fn hysteresis_stops_the_gate_flapping() {
        let mut g = Gate::new(GateConfig::default());
        feed(&mut g, &[0.9]);
        // Sitting between the two thresholds: still speech, and crucially the
        // silence timer stays at zero, so no end can ever fire here.
        let events = feed(&mut g, &[0.45; 40]);
        assert_eq!(events, vec![]);
        assert!(g.is_speech());
        assert_eq!(g.silence(), Duration::ZERO);
    }

    #[test]
    fn the_silence_timer_keeps_running_through_the_hold() {
        // `li-stream` flushes a sentence at pause_flush = 0.6s, which is longer
        // than min_silence. If the timer restarted when SpeechEnd fired, that
        // flush would never be reached.
        let mut g = Gate::new(GateConfig::default());
        feed(&mut g, &[0.9]);
        feed(&mut g, &[0.0; 25]);
        assert!(!g.is_speech());
        assert_eq!(g.silence(), 25 * W);
        assert!(g.silence() >= Duration::from_millis(600));
    }

    #[test]
    fn silence_is_zero_again_once_speech_resumes() {
        let mut g = Gate::new(GateConfig::default());
        feed(&mut g, &[0.9]);
        feed(&mut g, &[0.0; 10]);
        assert_eq!(feed(&mut g, &[0.9]), vec![VadEvent::SpeechStart]);
        assert_eq!(g.silence(), Duration::ZERO);
    }

    #[test]
    fn reset_clears_the_state() {
        let mut g = Gate::new(GateConfig::default());
        feed(&mut g, &[0.9, 0.0, 0.0]);
        g.reset();
        assert!(!g.is_speech());
        assert_eq!(g.silence(), Duration::ZERO);
        assert_eq!(g.probability(), 0.0);
    }
}
