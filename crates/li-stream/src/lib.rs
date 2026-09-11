//! Commit policy and dual-lane merge (PLAN §12).
//!
//! [`stream`] is the state machine: the fast lane opens and closes subtitle
//! lines, the accurate lane later replaces their text under the same id, and
//! [`merge`] decides which of the two a line ends up keeping. [`agree`] holds
//! what has been committed, and [`load`] watches what the accurate lane costs.
//!
//! Nothing here starts a thread, owns a model, or reads a clock. `li-core`
//! drives it.
//!
//! **The accurate lane does not re-transcribe.** PLAN §12.3 specified
//! LocalAgreement-2 over a rolling buffer; task 1.5 measured it against
//! transcribing one whole utterance at a time and it lost on every count --
//! accuracy, completeness and cost. See [`agree`] for the numbers.

use li_types::{FastReason, Lane};

pub mod agree;
pub mod load;
pub mod merge;
pub mod stream;
/// Word-level text helpers. Public because `li-core` needs the same idea of
/// what a filler is when it decides whether a line is worth translating -- a
/// line of nothing but "Okay." is 20.5% of them, and NLLB turns every one into
/// a hallucination (PLAN §19-20).
pub mod text;

pub use stream::{LaneMode, Stream};

/// Tuning that Phase 0 and the task 1.0 spikes settled on.
#[derive(Debug, Clone, Copy)]
pub struct StreamConfig {
    /// Trailing silence that ends a sentence, and the fast lane's line boundary.
    pub pause_flush_s: f64,
    /// Longest utterance handed to the accurate lane in one pass. A speaker who
    /// never pauses still has to reach the screen, and whisper's own input
    /// window is 30 s. Cutting here is a compromise -- the cut lands mid-clause
    /// and whisper is worse at those (see [`agree`]) -- so it is a limit, not a
    /// target: the endpoint detector normally gets there first.
    pub max_segment_s: f64,
    /// Promote the fast lane's text if the accurate lane has not delivered.
    pub promote_after_s: f64,
    /// Longest subtitle line, in words.
    ///
    /// High on purpose. One endpoint should be one line: splitting a fast-lane
    /// segment into several means the accurate lane's answer for that segment
    /// has to be sliced across them by timestamp, and the two engines do not
    /// agree closely enough about word times for that to be free (measured:
    /// 19.1% content WER at a 12-word cap against 18.7% uncut, on the same
    /// audio). This is the safety valve for sherpa's 20 s utterance rule; the
    /// display wraps.
    pub max_words: usize,
    /// Characters of committed text carried into whisper's `initial_prompt`.
    pub prompt_chars: usize,
    /// Reject accurate-lane text shorter than this fraction of the fast lane's
    /// *content* word count. See [`merge::decide`].
    pub truncation_guard: f64,
    /// ...and only when it is short by at least this many content words. Matches
    /// the definition of a hole in the G2b metric: a run of >= 5 deletions.
    /// Without it, a three-word line where one filler differs trips the guard.
    pub min_drop_words: usize,
    /// The mirror image: reject accurate-lane text longer than this multiple of
    /// the fast lane's content word count *and* repeating itself (task 1.4).
    /// Legitimately longer output is common -- whisper spells numbers out and
    /// hears words the fast lane misses -- so this sits well above 1.0 and is
    /// still only half the test.
    pub repetition_guard: f64,
    /// ...and only when it is longer by at least this many content words.
    pub min_extra_words: usize,
    /// Length of the repeated run that confirms a loop, in content words.
    pub repetition_run: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            pause_flush_s: 0.6,
            max_segment_s: 12.0,
            promote_after_s: 8.0,
            max_words: 40,
            prompt_chars: 200,
            truncation_guard: 0.6,
            min_drop_words: 5,
            repetition_guard: 1.5,
            min_extra_words: 5,
            repetition_run: 4,
        }
    }
}

/// What a line should finally say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub text: String,
    pub lane: Lane,
    pub reason: Option<FastReason>,
}
