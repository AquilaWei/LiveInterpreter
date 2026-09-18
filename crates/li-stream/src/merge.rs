//! Choosing between the two lanes for one subtitle line.
//!
//! The fast lane is the reference, not because it is more accurate -- it is
//! not, by 5-7 WER points -- but because its failure modes are *small*. It
//! mis-hears words. It never drops a clause and it never repeats one. Whisper
//! does both, and both are invisible to a WER number while being exactly the
//! failures that ruin a subtitle and corrupt the transcript file.
//! So the accurate lane's text is accepted unless it disagrees with the fast
//! lane about *how much was said*, in either direction.

use li_types::{FastReason, Lane};

use crate::text::{content_words, has_repeated_run};
use crate::{Decision, StreamConfig};

/// Decide what a line finally says.
///
/// `accurate` is `None` while that lane is still working, or forever if it is
/// switched off. `timed_out` means `promote_after_s` has elapsed.
///
/// The two interesting cases are the guards.
///
/// **Too short.** Whisper's decoder is a language model and condenses when it
/// is unsure, so it does not merely mis-hear -- it drops whole clauses. Task
/// Measured across six clips: `base.en` dropped nothing, but `small`
/// and `medium` each lost a run of five or more words on three of the six, up
/// to 20 words at once, while the fast lane never did. That defeats the timeout
/// rule, because the accurate lane *did* answer; it just answered short.
///
/// **Too long.** The same engine also gets stuck repeating a clause. A benchmark
/// reproduced it on demand and measured the damage (content WER 15.2% ->
/// 36.1%), and it is the mirror image: the output is longer, so a ratio guard
/// written only for the short case never fires. The engine-level fix is not to
/// crop whisper's encoder context, but that is a mitigation, not a
/// guarantee -- whisper loops occasionally anyway.
///
/// Both guards are measured in content words and both need an absolute
/// difference as well as a ratio: fillers disappearing between the lanes is
/// normal, and a three-word line must not trip either guard on one word.
pub fn decide(
    cfg: &StreamConfig,
    fast: Option<&str>,
    accurate: Option<&str>,
    timed_out: bool,
) -> Option<Decision> {
    match (fast, accurate) {
        (_, Some(acc)) => {
            let keep_fast = fast
                .and_then(|f| suspect(cfg, f, acc))
                .map(|reason| Decision {
                    text: fast.unwrap().to_owned(),
                    lane: Lane::Fast,
                    reason: Some(reason),
                });
            keep_fast.or(Some(Decision {
                text: acc.to_owned(),
                lane: Lane::Accurate,
                reason: None,
            }))
        }
        (Some(f), None) if timed_out => Some(Decision {
            text: f.to_owned(),
            lane: Lane::Fast,
            reason: Some(FastReason::AccurateTimeout),
        }),
        // Still waiting, or nothing was said at all.
        _ => None,
    }
}

/// `Some(reason)` when the accurate text disagrees with the fast lane about how
/// much was said by enough to be a dropped clause or a repetition loop.
fn suspect(cfg: &StreamConfig, fast: &str, accurate: &str) -> Option<FastReason> {
    let (f, a) = (content_words(fast), content_words(accurate));
    if f.is_empty() {
        return None;
    }
    let (fl, al) = (f.len(), a.len());

    // Nothing at all where the fast lane heard something is never an
    // improvement, however short the line: an empty override would blank the
    // subtitle and write an empty line into the transcript.
    if al == 0 {
        return Some(FastReason::AccurateTruncated);
    }
    if fl.saturating_sub(al) >= cfg.min_drop_words && (al as f64) < cfg.truncation_guard * fl as f64
    {
        return Some(FastReason::AccurateTruncated);
    }
    // A long line is not by itself wrong -- whisper spells numbers out, expands
    // contractions, and catches words the fast lane missed -- so the repetition
    // has to be there as well.
    if al.saturating_sub(fl) >= cfg.min_extra_words
        && (al as f64) > cfg.repetition_guard * fl as f64
        && has_repeated_run(&a, cfg.repetition_run)
    {
        return Some(FastReason::AccurateLooped);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAST: &str = "this is our first meeting surprisingly enough this is our agenda";

    fn cfg() -> StreamConfig {
        StreamConfig::default()
    }

    #[test]
    fn accurate_text_wins_when_it_is_complete() {
        let acc = "This is our first meeting, surprisingly enough. This is our agenda.";
        let d = decide(&cfg(), Some(FAST), Some(acc), false).unwrap();
        assert_eq!(d.lane, Lane::Accurate);
        assert_eq!(d.text, acc);
    }

    #[test]
    fn a_dropped_clause_does_not_overwrite_the_fast_lane() {
        // What small.en actually returned for this span: the
        // "first meeting surprisingly enough this is our" clause is gone.
        let d = decide(&cfg(), Some(FAST), Some("This is our agenda."), false).unwrap();
        assert_eq!(d.lane, Lane::Fast);
        assert_eq!(d.reason, Some(FastReason::AccurateTruncated));
        assert_eq!(d.text, FAST);
    }

    #[test]
    fn shorter_but_not_truncated_is_accepted() {
        // Fillers and backchannels legitimately disappear; that is not a hole.
        let d = decide(
            &cfg(),
            Some("um so this is our agenda okay"),
            Some("This is our agenda."),
            false,
        )
        .unwrap();
        assert_eq!(d.lane, Lane::Accurate);
    }

    #[test]
    fn a_repetition_loop_does_not_overwrite_the_fast_lane() {
        // Verbatim from a cropped-encoder run on `ami_meeting2`,
        // against the fast lane's text for the same span.
        let fast = "AND THEN WHEN YOU GO ON THE MENU YOU CAN SELECT THE SUMMARIZATION BOX";
        let acc = "And then when you go on the menu, you can select the description box, \
                   and then when you go on the menu, you can select the description box \
                   and then when you go on the menu, you can select the description box \
                   and then when you go on the menu, you can select the description";
        let d = decide(&cfg(), Some(fast), Some(acc), false).unwrap();
        assert_eq!(d.lane, Lane::Fast);
        assert_eq!(d.reason, Some(FastReason::AccurateLooped));
        assert_eq!(d.text, fast);
    }

    #[test]
    fn legitimately_longer_accurate_text_is_accepted() {
        // Whisper spells numbers out and catches words the fast lane missed.
        // Longer is normal; longer *and repeating itself* is not.
        let d = decide(
            &cfg(),
            Some("WE HAVE TWENTY FOUR OF THEM"),
            Some("Well, we have twenty-four of them in the second cabinet, I believe."),
            false,
        )
        .unwrap();
        assert_eq!(d.lane, Lane::Accurate);
    }

    #[test]
    fn a_short_line_repeating_a_word_is_not_a_loop() {
        let d = decide(&cfg(), Some("NO NO"), Some("No, no, no, no."), false).unwrap();
        assert_eq!(d.lane, Lane::Accurate);
    }

    #[test]
    fn fast_lane_is_promoted_once_the_accurate_lane_times_out() {
        let d = decide(&cfg(), Some(FAST), None, true).unwrap();
        assert_eq!(d.reason, Some(FastReason::AccurateTimeout));
    }

    #[test]
    fn nothing_is_emitted_while_the_accurate_lane_is_still_working() {
        assert!(decide(&cfg(), Some(FAST), None, false).is_none());
    }

    #[test]
    fn accurate_only_mode_needs_no_fast_lane() {
        let d = decide(&cfg(), None, Some("This is our agenda."), false).unwrap();
        assert_eq!(d.lane, Lane::Accurate);
    }
}
