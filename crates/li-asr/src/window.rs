//! The rolling audio history the accurate lane is fed out of.
//!
//! The fast lane decides where an utterance starts and ends (PLAN §12.2), and
//! the accurate lane is then handed exactly that stretch of the *raw* audio --
//! not the audio the fast lane consumed, which has already gone. So something
//! has to keep the recent past, and something has to cut a span out of it by
//! time rather than by sample count.
//!
//! Both the engine and `xtask eval` do this, and they must do it identically:
//! the harness measuring the accurate lane's WER has to hand it the same window
//! the product will. That is the whole reason this is a type and not two copies
//! of the same index arithmetic.

use std::collections::VecDeque;
use std::time::Duration;

const SAMPLE_RATE: f64 = 16_000.0;

/// Audio kept either side of an utterance before handing it over.
///
/// The two engines place a boundary slightly differently, and a clipped first
/// phoneme is a mis-heard word.
pub const PAD: Duration = Duration::from_millis(300);

/// How far back to keep raw audio. Comfortably more than
/// `StreamConfig::max_segment_s` plus the promotion timeout, so a span is
/// always still there when it is asked for.
pub const KEEP: Duration = Duration::from_secs(30);

/// A ring of recent audio, addressed by position on the audio timeline.
#[derive(Debug)]
pub struct Ring {
    pcm: VecDeque<f32>,
    /// Where `pcm[0]` sits on the audio timeline.
    origin: Duration,
    keep: usize,
}

impl Default for Ring {
    fn default() -> Self {
        Self::new(KEEP)
    }
}

impl Ring {
    pub fn new(keep: Duration) -> Self {
        Self {
            pcm: VecDeque::new(),
            origin: Duration::ZERO,
            keep: (keep.as_secs_f64() * SAMPLE_RATE) as usize,
        }
    }

    /// Append a frame and drop whatever has aged out.
    pub fn push(&mut self, pcm: &[f32]) {
        self.pcm.extend(pcm.iter().copied());
        if self.pcm.len() > self.keep {
            let drop = self.pcm.len() - self.keep;
            self.pcm.drain(..drop);
            self.origin += Duration::from_secs_f64(drop as f64 / SAMPLE_RATE);
        }
    }

    /// Where the oldest sample still held sits on the audio timeline.
    pub fn origin(&self) -> Duration {
        self.origin
    }

    pub fn len(&self) -> usize {
        self.pcm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pcm.is_empty()
    }

    /// The audio for `t_start..t_end`, padded by [`PAD`], and where the
    /// returned samples start on the audio timeline.
    ///
    /// Clamped to what is still held, so asking for a span that has aged out
    /// gives back the part that has not rather than an error -- the caller is
    /// mid-sentence and a short window beats no window.
    pub fn cut(&self, t_start: Duration, t_end: Duration) -> (Vec<f32>, Duration) {
        let a = t_start.saturating_sub(PAD);
        let b = t_end + PAD;
        let at = |t: Duration| (t.saturating_sub(self.origin).as_secs_f64() * SAMPLE_RATE) as usize;
        let (lo, hi) = (at(a).min(self.pcm.len()), at(b).min(self.pcm.len()));
        let pcm = self
            .pcm
            .iter()
            .copied()
            .skip(lo)
            .take(hi.saturating_sub(lo))
            .collect();
        (pcm, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(n: f64) -> Duration {
        Duration::from_secs_f64(n)
    }

    /// One second of a ramp, so a cut can be checked by its values.
    fn frame(from: f64, len: f64) -> Vec<f32> {
        let n = (len * SAMPLE_RATE) as usize;
        let base = (from * SAMPLE_RATE) as usize;
        (0..n).map(|i| (base + i) as f32).collect()
    }

    #[test]
    fn a_cut_is_padded_on_both_sides() {
        let mut r = Ring::default();
        r.push(&frame(0.0, 5.0));
        let (pcm, origin) = r.cut(secs(2.0), secs(3.0));
        assert_eq!(origin, secs(1.7));
        assert_eq!(pcm.len(), (1.6 * SAMPLE_RATE) as usize);
        assert_eq!(pcm[0], (1.7 * SAMPLE_RATE) as f32);
    }

    #[test]
    fn the_start_of_the_clip_cannot_be_padded_below_zero() {
        let mut r = Ring::default();
        r.push(&frame(0.0, 5.0));
        let (pcm, origin) = r.cut(secs(0.1), secs(1.0));
        assert_eq!(origin, Duration::ZERO);
        assert_eq!(pcm[0], 0.0);
        assert_eq!(pcm.len(), (1.3 * SAMPLE_RATE) as usize);
    }

    #[test]
    fn old_audio_ages_out_and_the_origin_moves_with_it() {
        let mut r = Ring::new(secs(2.0));
        r.push(&frame(0.0, 1.0));
        r.push(&frame(1.0, 1.0));
        r.push(&frame(2.0, 1.0));
        assert_eq!(r.origin(), secs(1.0));
        assert_eq!(r.len(), (2.0 * SAMPLE_RATE) as usize);
        // A span that is still held is still cut correctly after the shift.
        let (pcm, origin) = r.cut(secs(2.5), secs(2.6));
        assert_eq!(origin, secs(2.2));
        assert_eq!(pcm[0], (2.2 * SAMPLE_RATE) as f32);
    }

    #[test]
    fn a_span_that_has_aged_out_gives_back_what_is_left() {
        // Not an error: the caller is mid-sentence, and a short window beats
        // no window at all.
        let mut r = Ring::new(secs(1.0));
        r.push(&frame(0.0, 3.0));
        let (pcm, _) = r.cut(secs(0.0), secs(0.5));
        assert!(pcm.is_empty());
    }
}
