//! Is the accurate lane keeping up?
//!
//! The Phase 0 PoC collapsed because nothing was watching this. Streaming work
//! per wall second for a lane that re-transcribes a rolling buffer is
//! `RTF_offline * (buffer / tick)`, so the buffer length multiplies the cost and
//! an unbounded buffer feeds on itself: slower passes -> less audio released per
//! pass -> longer buffer -> slower passes. The PoC reached ~22 s and inferred a
//! 28x work factor (task 1.0b).
//!
//! Task 1.5 took the multiplier out rather than controlling it: the lane now
//! transcribes each utterance once, so work per audio second is just the
//! engine's RTF and the quantity that can run away is the *utterance* length,
//! which the endpoint detector bounds. Measured 0.07-0.16 against 0.41-0.45 for
//! the re-transcribing version. What is left to watch is whether this machine
//! is fast enough at all -- the answer to which is a smaller model (PLAN §8),
//! not a shorter window.

use std::collections::VecDeque;
use std::time::Duration;

/// Passes averaged over. Long enough not to react to one slow pass, short
/// enough to notice a machine that is genuinely falling behind.
const WINDOW: usize = 32;

/// Tracks what the accurate lane costs.
#[derive(Debug, Clone, Default)]
pub struct Load {
    passes: VecDeque<(Duration, Duration)>,
    sum_window: f64,
    sum_cost: f64,
}

impl Load {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one pass: the length of the utterance, and how long it took.
    pub fn observe(&mut self, window: Duration, cost: Duration) {
        if self.passes.len() == WINDOW
            && let Some((w, c)) = self.passes.pop_front()
        {
            self.sum_window -= w.as_secs_f64();
            self.sum_cost -= c.as_secs_f64();
        }
        self.passes.push_back((window, cost));
        self.sum_window += window.as_secs_f64();
        self.sum_cost += cost.as_secs_f64();
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Mean utterance length handed to the engine.
    pub fn mean_window(&self) -> Duration {
        Duration::from_secs_f64(self.sum_window / self.passes.len().max(1) as f64)
    }

    /// Engine seconds spent per second of audio. 1.0 is exactly keeping up.
    ///
    /// Not per second of *wall* time: with one pass per utterance the two are
    /// the same in the long run, and per audio second is the number that can be
    /// compared against the offline RTF task 1.0b measured.
    pub fn work_factor(&self) -> f64 {
        if self.sum_window <= 0.0 {
            return 0.0;
        }
        self.sum_cost / self.sum_window
    }

    /// Headroom is gone. `li-core` degrades to a smaller model rather than
    /// letting the lane fall behind the clock (PLAN §8, task 1.4).
    pub fn keeping_up(&self) -> bool {
        self.passes.is_empty() || self.work_factor() < 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64) -> Duration {
        Duration::from_secs_f64(x)
    }

    #[test]
    fn work_is_engine_time_per_second_of_audio() {
        let mut l = Load::new();
        for _ in 0..10 {
            l.observe(s(4.0), s(0.6));
        }
        assert!((l.work_factor() - 0.15).abs() < 1e-9);
        assert_eq!(l.mean_window(), s(4.0));
        assert!(l.keeping_up());
    }

    #[test]
    fn a_lane_slower_than_the_clock_is_not_keeping_up() {
        let mut l = Load::new();
        l.observe(s(4.0), s(5.0));
        assert!(!l.keeping_up());
    }

    #[test]
    fn nothing_measured_yet_is_not_a_failure() {
        assert!(Load::new().keeping_up());
        assert_eq!(Load::new().work_factor(), 0.0);
    }

    #[test]
    fn the_window_forgets_old_passes() {
        let mut l = Load::new();
        for _ in 0..WINDOW {
            l.observe(s(4.0), s(6.0));
        }
        for _ in 0..WINDOW {
            l.observe(s(4.0), s(0.4));
        }
        assert!((l.work_factor() - 0.1).abs() < 1e-9);
    }
}
