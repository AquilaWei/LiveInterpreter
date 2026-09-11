//! Downmix to mono and resample to 16 kHz, incrementally.
//!
//! Capture devices hand us 44.1 or 48 kHz stereo; every model downstream wants
//! 16 kHz mono. Dropping samples to get there aliases everything above 8 kHz
//! back down into the speech band, so the rate change is preceded by a low-pass
//! filter. The Phase 0 PoC skipped this because `parec` resampled for us.
//!
//! State is carried across calls -- filter history and the fractional read
//! position -- so a stream split into arbitrary chunks gives the same result as
//! the whole buffer at once. That is worth a test, because getting it wrong
//! produces a click at every chunk boundary rather than an obvious failure.

use std::collections::VecDeque;

/// Odd, so the filter has an exact centre tap.
const TAPS: usize = 63;
/// Slightly below Nyquist, leaving the transition band off the speech content.
const CUTOFF: f32 = 0.45;

#[derive(Debug, Clone)]
pub struct Resampler {
    ratio: f64,
    taps: Option<Vec<f32>>,
    history: VecDeque<f32>,
    /// Filtered samples not yet consumed, plus the one needed for interpolation.
    pending: Vec<f32>,
    pos: f64,
}

impl Resampler {
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        let taps =
            (dst_rate < src_rate).then(|| low_pass(CUTOFF * dst_rate as f32 / src_rate as f32));
        Self {
            ratio: src_rate as f64 / dst_rate as f64,
            taps,
            history: VecDeque::from(vec![0.0; TAPS - 1]),
            pending: Vec::new(),
            pos: 0.0,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.ratio == 1.0 && self.taps.is_none()
    }

    /// Interleaved `channels`-channel input -> mono at the destination rate.
    pub fn process(&mut self, input: &[f32], channels: u16) -> Vec<f32> {
        let mono = downmix(input, channels);
        if self.is_identity() {
            return mono;
        }
        match &self.taps {
            Some(taps) => {
                for &s in &mono {
                    self.history.push_back(s);
                    let acc: f32 = taps
                        .iter()
                        .zip(self.history.iter())
                        .map(|(t, h)| t * h)
                        .sum();
                    self.pending.push(acc);
                    self.history.pop_front();
                }
            }
            None => self.pending.extend_from_slice(&mono),
        }

        let mut out = Vec::new();
        while self.pos + 1.0 < self.pending.len() as f64 {
            let i = self.pos as usize;
            let f = (self.pos - i as f64) as f32;
            out.push(self.pending[i] * (1.0 - f) + self.pending[i + 1] * f);
            self.pos += self.ratio;
        }
        // `pos` can land past the end when the ratio is greater than one: the
        // next output sample simply falls inside the chunk that has not arrived
        // yet. Clamp the drain and keep the overshoot in `pos`, which then
        // reads as "skip this many of the samples that come next".
        let consumed = (self.pos as usize).min(self.pending.len());
        self.pending.drain(..consumed);
        self.pos -= consumed as f64;
        out
    }
}

fn downmix(input: &[f32], channels: u16) -> Vec<f32> {
    match channels {
        0 | 1 => input.to_vec(),
        n => {
            let n = n as usize;
            input
                .chunks_exact(n)
                .map(|f| f.iter().sum::<f32>() / n as f32)
                .collect()
        }
    }
}

/// Windowed-sinc low-pass. `cutoff` is a fraction of the source sample rate.
fn low_pass(cutoff: f32) -> Vec<f32> {
    let mid = (TAPS / 2) as f32;
    let mut taps: Vec<f32> = (0..TAPS)
        .map(|i| {
            let x = i as f32 - mid;
            let sinc = if x == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * std::f32::consts::PI * cutoff * x).sin() / (std::f32::consts::PI * x)
            };
            // Hann window
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (TAPS - 1) as f32).cos();
            sinc * w
        })
        .collect();
    let sum: f32 = taps.iter().sum();
    for t in &mut taps {
        *t /= sum;
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, freq: f32, rate: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        assert_eq!(downmix(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
    }

    #[test]
    fn matching_rates_pass_through_untouched() {
        let mut r = Resampler::new(16_000, 16_000);
        let x = sine(256, 440.0, 16_000);
        assert_eq!(r.process(&x, 1), x);
    }

    #[test]
    fn output_length_follows_the_rate_ratio() {
        let mut r = Resampler::new(48_000, 16_000);
        let out = r.process(&sine(4800, 440.0, 48_000), 1);
        assert!((out.len() as i64 - 1600).abs() <= 2, "got {}", out.len());
    }

    #[test]
    fn chunked_input_gives_the_same_result_as_one_buffer() {
        let x = sine(9600, 440.0, 48_000);
        let whole = Resampler::new(48_000, 16_000).process(&x, 1);
        let mut r = Resampler::new(48_000, 16_000);
        let mut chunked = Vec::new();
        for c in x.chunks(317) {
            chunked.extend(r.process(c, 1));
        }
        assert_eq!(whole.len(), chunked.len());
        for (a, b) in whole.iter().zip(&chunked) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn content_above_the_new_nyquist_is_filtered_out() {
        // 14 kHz cannot exist at 16 kHz; without the low-pass it would alias
        // back to 2 kHz, right in the middle of speech.
        let mut r = Resampler::new(48_000, 16_000);
        let out = r.process(&sine(24_000, 14_000.0, 48_000), 1);
        let level = out[out.len() / 2..]
            .iter()
            .fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(level < 0.05, "aliased energy {level}");
    }

    #[test]
    fn speech_band_content_survives() {
        let mut r = Resampler::new(48_000, 16_000);
        let out = r.process(&sine(24_000, 440.0, 48_000), 1);
        let level = out[out.len() / 2..]
            .iter()
            .fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(level > 0.9, "lost the signal: {level}");
    }
}
