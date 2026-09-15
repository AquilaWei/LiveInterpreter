//! When a restored full stop is worth ending a line on (PLAN task 1.25, S2).
//!
//! The fast lane's endpoint detector asks the audio: 0.6 s of silence closes a
//! line. Task 1.24 measured what happens when that number is lowered and the
//! answer was a clear no -- conversation hesitates mid-sentence, and
//! `ami_meeting`'s content WER went from 14.7% to 21.9%. So the acoustic route
//! is finished, and this is the other one: ask the *words* where the sentence
//! ended.
//!
//! Nothing here loads a model. `li-core` owns the punctuation model and hands
//! this module the text it produced, which keeps the crate's one rule intact --
//! no threads, no models, no clocks -- and keeps the policy testable without
//! an ONNX session. `now` is audio time, passed in, as everywhere else here.
//!
//! # The rule, and where each number came from
//!
//! Task 1.25's E2 ran the punctuation model over ~960 fast-lane partials across
//! six clips and watched what the marks did before the line closed. Three of
//! the four conditions below are read off that measurement; the fourth is not,
//! and says so.
//!
//! 1. **Two consecutive runs.** A mark has to survive one more hypothesis
//!    before it counts. E2 measured the cost at 0.32-1.6 s of the lead time and
//!    the benefit as nearly total: flaps before close came out at a median of
//!    **0** and a maximum of **2**, so on a decoded prefix the punctuator is
//!    very nearly deterministic and one extra run buys almost all of it.
//! 2. **At least one word after it.** A full stop on the last decoded word is
//!    the model guessing the utterance is over; the same mark with a word
//!    already behind it is the model having seen what came next and kept it.
//!    This condition is also the only reason a real `t_end` exists -- the
//!    boundary is the *next* word's onset, and that word has to have been
//!    decoded for anyone to know when it starts. Correctness and confidence
//!    turn out to be the same condition, which is the sign the shape is right.
//! 3. **Enough words to be a line.** Matches `xtask pause --min-words 4`: a
//!    two-word fragment on its own row is worse than the wait it saved.
//! 4. **A gap since the last boundary.** This one is a guard, not a
//!    measurement. Without it the punctuator can cut at word 20 and the
//!    endpoint detector close the rest 200 ms later, and the two lines then
//!    fight over one row under the bar's `min_dwell_ms = 700`. The value is
//!    picked from that dwell with headroom; E2 says nothing about it, and if it
//!    ever needs tuning it needs its own measurement.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::text;

/// How hard a restored mark has to work before a line ends on it.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryConfig {
    /// Consecutive hypotheses a boundary must appear in. See rule 1.
    pub confirm_runs: u32,
    /// Words that must already be decoded after it. See rule 2.
    pub min_words_after: usize,
    /// Content words the line being closed must have. See rule 3.
    pub min_line_words: usize,
    /// Audio time since the last boundary of any kind. See rule 4.
    pub min_gap_s: f64,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            confirm_runs: 2,
            min_words_after: 1,
            min_line_words: 4,
            min_gap_s: 1.0,
        }
    }
}

/// Watches one open line's hypotheses for a sentence end worth cutting on.
#[derive(Debug)]
pub struct BoundaryPolicy {
    cfg: BoundaryConfig,
    /// Boundary index -> how many consecutive runs it has held.
    seen: BTreeMap<usize, u32>,
    /// Words of the open segment already emitted as their own line, if any.
    /// The next line starts here, and only a boundary past it can be cut.
    emitted: usize,
    /// Audio time of the last boundary, from either source.
    last: Duration,
}

impl BoundaryPolicy {
    pub fn new(cfg: BoundaryConfig) -> Self {
        Self {
            cfg,
            seen: BTreeMap::new(),
            emitted: 0,
            last: Duration::ZERO,
        }
    }

    /// A new hypothesis for the open line. `text` is punctuated; `now` is where
    /// the audio has reached.
    ///
    /// `Some(i)` means the line should end on word `i` (inclusive, counted from
    /// the start of the open segment). The caller still has to find a real
    /// onset for word `i + 1` before it may act, and may decline: nothing here
    /// changes until [`Self::cut`] says it did.
    pub fn observe(&mut self, text: &str, now: Duration) -> Option<usize> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let ends: Vec<usize> = words
            .iter()
            .enumerate()
            .filter(|(_, w)| ends_sentence(w))
            .map(|(i, _)| i)
            .collect();

        // A boundary that went away starts over. `retain` before the bump so a
        // mark present in this run is never punished for this run.
        self.seen.retain(|i, _| ends.contains(i));
        for i in &ends {
            *self.seen.entry(*i).or_default() += 1;
        }

        if now < self.last + Duration::from_secs_f64(self.cfg.min_gap_s) {
            return None;
        }
        // Latest first: a line that ends on the last sentence available is one
        // line, where taking the earliest would leave the rest to be cut again.
        self.seen
            .iter()
            .rev()
            .filter(|(_, runs)| **runs >= self.cfg.confirm_runs)
            .map(|(i, _)| *i)
            .find(|i| self.worth_cutting(&words, *i))
    }

    fn worth_cutting(&self, words: &[&str], i: usize) -> bool {
        i >= self.emitted
            && i + self.cfg.min_words_after < words.len()
            && text::content_words(&words[self.emitted..=i].join(" ")).len()
                >= self.cfg.min_line_words
    }

    /// Words of the open segment that are already lines of their own. The
    /// caller shows the hypothesis from here on, so the bar does not repeat a
    /// sentence it has already settled.
    pub fn emitted(&self) -> usize {
        self.emitted
    }

    /// The caller acted on `observe`: words `..=i` are now a line of their own.
    pub fn cut(&mut self, i: usize, now: Duration) {
        self.emitted = i + 1;
        self.last = now;
        self.seen.clear();
    }

    /// The segment closed on its own. The next hypothesis is a fresh line.
    pub fn settle(&mut self, now: Duration) {
        self.emitted = 0;
        self.last = now;
        self.seen.clear();
    }
}

/// Does this word carry a sentence-ending mark?
///
/// Closing quotes and brackets come after the stop, so they are trimmed first.
/// The restored marks are the model's, not the speaker's, which is exactly why
/// a mark alone is not enough -- see the rules above.
fn ends_sentence(w: &str) -> bool {
    w.trim_end_matches(['"', '\'', ')', ']', '\u{201d}'])
        .ends_with(['.', '!', '?'])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(x: f64) -> Duration {
        Duration::from_secs_f64(x)
    }

    /// Past the `min_gap_s` guard, which is not what any of these test.
    fn policy() -> BoundaryPolicy {
        let mut p = BoundaryPolicy::new(BoundaryConfig::default());
        p.settle(Duration::ZERO);
        p
    }

    /// Feed a run of hypotheses at one-second intervals from t=10.
    fn run(p: &mut BoundaryPolicy, hyps: &[&str]) -> Vec<Option<usize>> {
        hyps.iter()
            .enumerate()
            .map(|(i, h)| p.observe(h, t(10.0 + i as f64)))
            .collect()
    }

    #[test]
    fn a_mark_has_to_survive_one_more_hypothesis() {
        let mut p = policy();
        let seen = run(
            &mut p,
            &[
                "we can ship it on",
                "we can ship it. On",
                "we can ship it. On monday",
            ],
        );
        // Nothing on the first sighting; nothing on the second either, because
        // the mark is still on the last decoded word.
        assert_eq!(seen, [None, None, Some(3)]);
    }

    #[test]
    fn a_mark_on_the_last_word_is_the_model_guessing() {
        let mut p = policy();
        // Held for three runs, but never with a word behind it.
        assert_eq!(
            run(
                &mut p,
                &["we can ship it.", "we can ship it.", "we can ship it."]
            ),
            [None, None, None]
        );
    }

    #[test]
    fn a_mark_that_goes_away_starts_over() {
        let mut p = policy();
        let seen = run(
            &mut p,
            &[
                "we can ship it. On",
                "we can ship it on the",
                "we can ship it. On the",
                "we can ship it. On the first",
            ],
        );
        assert_eq!(seen, [None, None, None, Some(3)]);
    }

    #[test]
    fn a_fragment_is_not_worth_a_line() {
        let mut p = policy();
        // "Yeah." is one word and a filler at that: two ways of being too thin.
        assert_eq!(
            run(&mut p, &["yeah. We", "yeah. We can", "yeah. We can ship"]),
            [None, None, None]
        );
    }

    #[test]
    fn the_line_that_is_cut_is_the_last_sentence_available() {
        let mut p = policy();
        let hyp = "we can ship it. The build is green. On monday";
        run(&mut p, &[hyp, hyp]);
        // Not word 3: cutting at the earlier stop would leave the later one to
        // be cut again, turning one decision into two lines.
        assert_eq!(p.observe(hyp, t(20.0)), Some(7));
    }

    #[test]
    fn a_boundary_already_emitted_is_not_cut_twice() {
        let mut p = policy();
        let hyp = "we can ship it. The build is green";
        run(&mut p, &[hyp, hyp]);
        assert_eq!(p.observe(hyp, t(20.0)), Some(3));
        p.cut(3, t(20.0));

        // Same mark, more words behind it, and the gap has passed.
        assert_eq!(
            p.observe("we can ship it. The build is green now", t(30.0)),
            None
        );
    }

    #[test]
    fn the_second_line_of_a_segment_is_measured_from_the_first_cut() {
        let mut p = policy();
        let hyp = "we can ship it. The build is green";
        run(&mut p, &[hyp, hyp]);
        p.cut(3, t(20.0));

        // "The build is green." is four content words past the cut, so it
        // stands on its own once a word follows it.
        let hyp2 = "we can ship it. The build is green. Ship";
        assert_eq!(p.observe(hyp2, t(30.0)), None);
        assert_eq!(p.observe(hyp2, t(31.0)), Some(7));
    }

    #[test]
    fn nothing_is_cut_within_the_gap_of_the_last_boundary() {
        let mut p = BoundaryPolicy::new(BoundaryConfig::default());
        p.settle(t(10.0));
        let hyp = "we can ship it. The build is green";
        assert_eq!(p.observe(hyp, t(10.4)), None);
        assert_eq!(p.observe(hyp, t(10.8)), None);
        // Same evidence, only later.
        assert_eq!(p.observe(hyp, t(11.2)), Some(3));
    }

    #[test]
    fn a_closing_quote_does_not_hide_the_stop() {
        let mut p = policy();
        let hyp = "he said \"we can ship it.\" The build";
        run(&mut p, &[hyp, hyp]);
        assert_eq!(p.observe(hyp, t(20.0)), Some(5));
    }
}
