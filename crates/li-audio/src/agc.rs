//! Streaming input-level normalisation.
//!
//! Not cosmetic. Task 1.0a found the user's own meeting recording sat about
//! 11 dB below LibriSpeech, and on a level-sensitive streaming model that alone
//! took recognition from 168 words to 16 over the same two minutes. Corpus audio
//! is mastered; a microphone in a room or a loopback tap from a quiet player is
//! not, and the spread between them is far wider than between two corpora.
//!
//! It has to work forwards only -- there is no future audio to normalise
//! against -- so the level estimate is an exponential moving average of frame
//! RMS that adapts on speech and holds still on silence. Adapting on silence
//! would drive the estimate towards zero and then amplify room noise into the
//! recogniser between sentences.

/// Roughly LibriSpeech's level; the models in use were trained on corpora at
/// about this loudness.
const TARGET_RMS: f32 = 0.067;
/// Below this a frame is silence, and neither adapts the estimate nor is scaled.
const FLOOR_RMS: f32 = 1e-3;
const MAX_GAIN: f32 = 20.0;
/// Slow enough to ride out one loud syllable, fast enough to follow a speaker
/// leaning away from the microphone.
const SMOOTHING: f32 = 0.95;

#[derive(Debug, Clone)]
pub struct Agc {
    target_rms: f32,
    max_gain: f32,
    ema_rms: f32,
}

impl Default for Agc {
    fn default() -> Self {
        Self {
            target_rms: TARGET_RMS,
            max_gain: MAX_GAIN,
            ema_rms: 0.0,
        }
    }
}

impl Agc {
    /// Scale one frame in place. Only ever amplifies: audio already at or above
    /// the target is left alone rather than turned down, because the models
    /// tolerate loud input and attenuating would throw away headroom the
    /// recogniser is happy to have.
    pub fn process(&mut self, pcm: &mut [f32]) {
        let rms = rms(pcm);
        if rms > FLOOR_RMS {
            self.ema_rms = if self.ema_rms == 0.0 {
                rms
            } else {
                SMOOTHING * self.ema_rms + (1.0 - SMOOTHING) * rms
            };
        }
        if self.ema_rms <= FLOOR_RMS {
            return;
        }
        let gain = (self.target_rms / self.ema_rms).min(self.max_gain);
        if gain <= 1.0 {
            return;
        }
        for s in pcm {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }

    /// The current level estimate, for logging and the status UI.
    pub fn level(&self) -> f32 {
        self.ema_rms
    }
}

fn rms(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    let sum: f64 = pcm.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum / pcm.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| amp * (i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn quiet_input_is_brought_up_to_the_target() {
        let mut agc = Agc::default();
        let mut last = 0.0;
        // Feed the same quiet signal repeatedly; the estimate converges.
        for _ in 0..200 {
            let mut f = tone(512, 0.01);
            agc.process(&mut f);
            last = rms(&f);
        }
        assert!(
            (last - TARGET_RMS).abs() < TARGET_RMS * 0.15,
            "expected ~{TARGET_RMS}, got {last}"
        );
    }

    #[test]
    fn loud_input_is_left_alone() {
        let mut agc = Agc::default();
        let loud = tone(512, 0.5);
        let mut f = loud.clone();
        for _ in 0..50 {
            f = loud.clone();
            agc.process(&mut f);
        }
        assert_eq!(f, loud);
    }

    #[test]
    fn silence_does_not_move_the_estimate() {
        let mut agc = Agc::default();
        for _ in 0..100 {
            agc.process(&mut tone(512, 0.02));
        }
        let speech_level = agc.level();
        for _ in 0..500 {
            agc.process(&mut vec![0.0; 512]);
        }
        assert_eq!(agc.level(), speech_level);
    }

    #[test]
    fn a_silent_stream_is_never_amplified() {
        let mut agc = Agc::default();
        let mut f = vec![0.0f32; 512];
        agc.process(&mut f);
        assert!(f.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn output_never_clips() {
        let mut agc = Agc::default();
        for _ in 0..300 {
            let mut f = tone(512, 0.001);
            agc.process(&mut f);
            assert!(f.iter().all(|s| s.abs() <= 1.0));
        }
    }
}
