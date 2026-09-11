//! The dual-lane state machine (PLAN §12.2).
//!
//! One subtitle line has one `line_id` for its whole life. The fast lane opens
//! it and keeps rewriting it while the words are still arriving; the accurate
//! lane later replaces the text in place, under the same id, so the UI swaps a
//! line rather than appending one. Whichever lane ends up owning the line, its
//! `[t_start, t_end)` was fixed when the fast lane closed it -- which is also
//! why sentence spans come out right here, where the Phase 0 PoC wrote the same
//! timestamp into both ends and produced zero-length SRT blocks.
//!
//! **There are no clocks in this file.** Every entry point takes `now` as a
//! position on the audio timeline. That is not only for testing: audio arrives
//! in real time, so the audio clock advances at wall-clock rate, and when the
//! machine falls behind it is the audio clock that keeps going while the engine
//! does not -- exactly when the 8 s promotion rule should fire.

use std::collections::VecDeque;
use std::time::Duration;

use li_types::{EngineEvent, FastReason, Lane, Word};

use crate::agree::Committed;
use crate::{StreamConfig, merge, text};

/// Which lanes are running. Both single-lane modes are supported settings, not
/// degraded states: a machine with no GPU turns the accurate lane off, and a
/// phone on battery runs the fast lane alone (PLAN §8.2, §12.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneMode {
    Dual,
    /// No accurate lane. The transcript and the translator take fast-lane text,
    /// marked as such.
    FastOnly,
    /// No fast lane: Phase 0 behaviour, at Phase 0 latency (~2 s).
    AccurateOnly,
}

#[derive(Debug, Clone)]
struct Line {
    id: u64,
    t_start: Duration,
    t_end: Duration,
    fast: Option<String>,
    accurate: Option<String>,
}

pub struct Stream {
    cfg: StreamConfig,
    mode: LaneMode,
    next_id: u64,
    /// The line the fast lane's partials are currently going to, if any.
    open: Option<u64>,
    pending: VecDeque<Line>,
    committed: Committed,
    /// Committed accurate-lane words not yet claimed by a line.
    unrouted: Vec<Word>,
    /// Everything before this the accurate lane has said all it is going to say
    /// about, because it hit an end of utterance there.
    flushed_through: Duration,
    /// Everything before this has been emitted as final; late accurate words
    /// landing there belong to a line that is already on screen and in the
    /// transcript, and must not be re-attributed to the next one.
    emitted_through: Duration,
    now: Duration,
}

impl Stream {
    pub fn new(cfg: StreamConfig, mode: LaneMode) -> Self {
        Self {
            cfg,
            mode,
            next_id: 1,
            open: None,
            pending: VecDeque::new(),
            committed: Committed::new(),
            unrouted: Vec::new(),
            flushed_through: Duration::ZERO,
            emitted_through: Duration::ZERO,
            now: Duration::ZERO,
        }
    }

    pub fn mode(&self) -> LaneMode {
        self.mode
    }

    /// The accurate lane's commit frontier: where its audio buffer may be
    /// trimmed to. Feeds [`crate::buffer::BufferPolicy::plan`].
    pub fn frontier(&self) -> Duration {
        self.committed.frontier()
    }

    /// Text to carry into whisper's `initial_prompt`.
    pub fn prompt_tail(&self) -> String {
        self.committed.prompt_tail(self.cfg.prompt_chars)
    }

    /// Fast-lane hypothesis, still being revised. Screen only.
    pub fn fast_partial(&mut self, text: &str, now: Duration) -> Vec<EngineEvent> {
        self.at(now);
        let line_id = match self.open {
            Some(id) => id,
            None => {
                let id = self.alloc();
                self.open = Some(id);
                id
            }
        };
        let mut out = vec![EngineEvent::SourcePartial {
            line_id,
            text: text.to_owned(),
            lane: Lane::Fast,
            closed: None,
        }];
        out.extend(self.drain());
        out
    }

    /// The fast lane reached an endpoint: this is where a line boundary goes,
    /// and where the line's span on the audio timeline is fixed.
    ///
    /// A single endpoint can still be too long for a two-line subtitle bar --
    /// sherpa's own rule lets an utterance run to 12 s -- so a long run is cut
    /// into several lines by word count.
    ///
    /// Not by punctuation, even though task 1.25 means these words may now
    /// carry some. Cutting here is free of consequences only because every
    /// piece of one endpoint is emitted in the same instant, into a span the
    /// accurate lane has already been handed whole; a cut that moved a line
    /// *boundary* would make the accurate lane's one answer be sliced across
    /// two lines by timestamp, which is measured and not free (see the crate
    /// docs: 19.1% content WER at a 12-word cap against 18.7% uncut). That is
    /// task 1.25's stage 2, behind its own switch and its own measurements --
    /// not something to slip in here because the marks happen to have arrived.
    pub fn fast_final(&mut self, words: &[Word], now: Duration) -> Vec<EngineEvent> {
        self.at(now);
        let mut out = Vec::new();
        for chunk in words.chunks(self.cfg.max_words.max(1)) {
            let Some((first, last)) = chunk.first().zip(chunk.last()) else {
                continue;
            };
            let text = text::join(chunk);
            if text.is_empty() {
                continue;
            }
            let id = self.open.take().unwrap_or_else(|| {
                let id = self.next_id;
                self.next_id += 1;
                id
            });
            let t_end = last.end.max(first.start);
            self.pending.push_back(Line {
                id,
                t_start: first.start,
                t_end,
                fast: Some(text.clone()),
                accurate: None,
            });
            if self.mode != LaneMode::FastOnly {
                // Show the completed fast-lane sentence while the accurate lane
                // works on it; without this the line keeps whatever half-finished
                // partial arrived last.
                out.push(EngineEvent::SourcePartial {
                    line_id: id,
                    text,
                    lane: Lane::Fast,
                    // The fast lane is done with this line -- the words will
                    // not change again, only be replaced wholesale by the
                    // accurate lane. That is what makes it safe to translate
                    // as a draft, and the span is what dates it; see
                    // `EngineEvent::SourcePartial`.
                    closed: Some(t_end),
                });
            }
        }
        self.open = None;
        out.extend(self.drain());
        out
    }

    /// The accurate lane's finished answer for one whole utterance.
    ///
    /// The dual-lane path. `now` is where that utterance's audio ends; every
    /// line inside it is settled by this call, so a line is overwritten one
    /// pass after the fast lane closed it rather than after a re-transcription
    /// cycle. See [`crate::agree::Agreement::adopt`] for why there is nothing
    /// to agree with.
    pub fn accurate_segment(&mut self, words: &[Word], now: Duration) -> Vec<EngineEvent> {
        self.at(now);
        let newly = self.committed.adopt(words);
        self.absorb(newly);
        self.flushed_through = self.now;
        if self.mode == LaneMode::AccurateOnly {
            self.cut_accurate();
        }
        self.drain()
    }

    /// Advance the clock with no new recognition. Call this on a timer: the 8 s
    /// promotion must fire when the accurate lane has gone *quiet*, which is
    /// precisely when no event would otherwise arrive to drive it.
    pub fn tick(&mut self, now: Duration) -> Vec<EngineEvent> {
        self.at(now);
        self.drain()
    }

    fn at(&mut self, now: Duration) {
        self.now = self.now.max(now);
    }

    fn alloc(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn absorb(&mut self, newly: Vec<Word>) {
        let through = self.emitted_through;
        self.unrouted
            .extend(newly.into_iter().filter(|w| midpoint(w) >= through));
    }

    /// In accurate-only mode there is no fast lane to allocate line ids, so
    /// each utterance becomes a line (cut further only if it is very long).
    fn cut_accurate(&mut self) {
        let max = self.cfg.max_words.max(1);
        loop {
            let n = self.unrouted.len();
            let take = if n == 0 { return } else { n.min(max) };
            let chunk: Vec<Word> = self.unrouted.drain(..take).collect();
            let (first, last) = (&chunk[0], &chunk[chunk.len() - 1]);
            let (t_start, t_end) = (first.start, last.end.max(first.start));
            let id = self.alloc();
            self.pending.push_back(Line {
                id,
                t_start,
                t_end,
                fast: None,
                accurate: Some(text::join(&chunk)),
            });
        }
    }

    /// Emit every line at the front of the queue that has an answer.
    ///
    /// Strictly front-to-back, stopping at the first line that is still
    /// waiting: lines reach the transcript file in order, and a line that never
    /// gets accurate text blocks on its own timeout rather than letting the one
    /// behind it overtake.
    fn drain(&mut self) -> Vec<EngineEvent> {
        let promote_after = Duration::from_secs_f64(self.cfg.promote_after_s);
        let mut out = Vec::new();
        while let Some(line) = self.pending.front_mut() {
            // Covered means the accurate lane will not add anything else inside
            // this line: either it has committed past the end, or it hit an end
            // of utterance there. The second case is not a formality -- its last
            // word routinely ends a little before the fast lane's, and waiting
            // for the frontier alone would let every sentence-final line time
            // out and throw away perfectly good punctuated text.
            let covered =
                self.committed.frontier() >= line.t_end || self.flushed_through >= line.t_end;
            if self.mode == LaneMode::Dual && line.accurate.is_none() && covered {
                let claimed = claim(&mut self.unrouted, line.t_end);
                line.accurate = Some(text::join(&claimed));
            }

            let line = self.pending.front().expect("checked above");
            let timed_out = self.now >= line.t_end + promote_after;
            let decision = if self.mode == LaneMode::FastOnly {
                line.fast.clone().map(|text| crate::Decision {
                    text,
                    lane: Lane::Fast,
                    reason: Some(FastReason::AccurateDisabled),
                })
            } else {
                merge::decide(
                    &self.cfg,
                    line.fast.as_deref(),
                    line.accurate.as_deref(),
                    timed_out,
                )
            };
            let Some(d) = decision else { break };

            let line = self.pending.pop_front().expect("checked above");
            self.emitted_through = self.emitted_through.max(line.t_end);
            // A promoted line's accurate words may still be on their way. They
            // describe audio that is already on screen and in the transcript,
            // so they must not be handed to the line behind it.
            claim(&mut self.unrouted, self.emitted_through);
            out.push(EngineEvent::SourceFinal {
                line_id: line.id,
                text: d.text,
                t_start: line.t_start,
                t_end: line.t_end,
                lane: d.lane,
                reason: d.reason,
            });
        }
        out
    }
}

fn midpoint(w: &Word) -> Duration {
    w.start + (w.end.saturating_sub(w.start)) / 2
}

/// Take the words whose midpoint falls before `t_end`.
///
/// Alignment is by audio time, never by matching text: the two lanes disagree
/// about the words, which is the whole reason both exist (PLAN §12.2). Taking
/// from the front also sweeps up anything earlier that no line claimed, so a
/// stray word cannot sit in front of the queue forever.
fn claim(words: &mut Vec<Word>, t_end: Duration) -> Vec<Word> {
    let n = words.iter().take_while(|w| midpoint(w) < t_end).count();
    words.drain(..n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(x: f64) -> Duration {
        Duration::from_secs_f64(x)
    }

    fn words(spec: &[(&str, f64, f64)]) -> Vec<Word> {
        spec.iter()
            .map(|(text, s, e)| Word {
                text: (*text).into(),
                start: t(*s),
                end: t(*e),
            })
            .collect()
    }

    /// A run of words, evenly spaced, starting at `from`.
    fn say(text: &str, from: f64, per: f64) -> Vec<Word> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, w)| Word {
                text: w.into(),
                start: t(from + i as f64 * per),
                end: t(from + (i as f64 + 1.0) * per),
            })
            .collect()
    }

    #[derive(Debug, PartialEq)]
    struct Final {
        id: u64,
        text: String,
        lane: Lane,
        reason: Option<FastReason>,
        span: (f64, f64),
    }

    fn finals(events: &[EngineEvent]) -> Vec<Final> {
        events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::SourceFinal {
                    line_id,
                    text,
                    t_start,
                    t_end,
                    lane,
                    reason,
                } => Some(Final {
                    id: *line_id,
                    text: text.clone(),
                    lane: *lane,
                    reason: *reason,
                    span: (t_start.as_secs_f64(), t_end.as_secs_f64()),
                }),
                _ => None,
            })
            .collect()
    }

    fn partials(events: &[EngineEvent]) -> Vec<(u64, String, Lane)> {
        events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::SourcePartial {
                    line_id,
                    text,
                    lane,
                    ..
                } => Some((*line_id, text.clone(), *lane)),
                _ => None,
            })
            .collect()
    }

    fn settled_flags(events: &[EngineEvent]) -> Vec<bool> {
        events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::SourcePartial { closed, .. } => Some(closed.is_some()),
                _ => None,
            })
            .collect()
    }

    fn dual() -> Stream {
        Stream::new(StreamConfig::default(), LaneMode::Dual)
    }

    fn settle(s: &mut Stream, hyp: &[Word], now: f64) -> Vec<EngineEvent> {
        s.accurate_segment(hyp, t(now))
    }

    const FAST: &str = "THIS IS OUR AGENDA";
    const ACC: &str = "This is our agenda.";

    #[test]
    fn a_line_keeps_one_id_from_the_first_partial_to_the_accurate_override() {
        // The UI requirement behind this: the accurate lane replaces the line
        // in place. A new id would push a second line onto the bar (PLAN §12.2).
        let mut s = dual();
        let ev = s.fast_partial("THIS IS", t(0.6));
        assert_eq!(partials(&ev), [(1, "THIS IS".into(), Lane::Fast)]);

        let ev = s.fast_final(&say(FAST, 0.0, 0.35), t(1.4));
        assert_eq!(partials(&ev), [(1, FAST.into(), Lane::Fast)]);
        assert!(finals(&ev).is_empty(), "the accurate lane has not answered");

        let ev = settle(&mut s, &say(ACC, 0.0, 0.35), 3.0);
        let out = finals(&ev);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 1);
        assert_eq!(out[0].lane, Lane::Accurate);
        assert_eq!(out[0].text, ACC);
    }

    #[test]
    fn only_the_fast_lanes_endpoint_marks_a_partial_settled() {
        // `li-core` translates a settled partial as a draft, half a second
        // before the accurate lane finalises the line. A revisable hypothesis
        // must never be marked: they arrive dozens of times a second, and each
        // one would be an NLLB pass over text about to be replaced.
        let mut s = dual();
        assert_eq!(settled_flags(&s.fast_partial("THIS IS", t(0.6))), [false]);
        assert_eq!(
            settled_flags(&s.fast_final(&say(FAST, 0.0, 0.35), t(1.4))),
            [true]
        );
    }

    #[test]
    fn the_span_is_the_fast_lanes_and_the_override_does_not_move_it() {
        // Phase 0 wrote the same timestamp into both ends of every sentence and
        // produced zero-length SRT blocks. The span is fixed when the fast lane
        // closes the line; the accurate lane only changes the words.
        let mut s = dual();
        s.fast_final(
            &words(&[("THIS", 0.20, 0.60), ("AGENDA", 0.60, 1.40)]),
            t(1.4),
        );
        // Note the accurate lane's last word ends before the fast lane's: the
        // two engines time the same audio differently, and it is the endpoint,
        // not the word times, that says the line is done.
        let ev = settle(
            &mut s,
            &words(&[("This", 0.31, 0.55), ("agenda.", 0.55, 1.31)]),
            3.0,
        );
        let out = finals(&ev);
        assert_eq!(out[0].span, (0.20, 1.40));
        assert_eq!(out[0].lane, Lane::Accurate);
    }

    #[test]
    fn an_accurate_lane_that_goes_quiet_is_overtaken_after_eight_seconds() {
        // Nothing else would drive this: the rule has to fire precisely when no
        // accurate-lane event is arriving, which is why `tick` exists.
        let mut s = dual();
        s.fast_final(&say(FAST, 0.0, 0.35), t(1.4));
        assert!(finals(&s.tick(t(9.0))).is_empty());

        let out = finals(&s.tick(t(9.5)));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].lane, Lane::Fast);
        assert_eq!(out[0].reason, Some(FastReason::AccurateTimeout));
        assert_eq!(out[0].text, FAST);
    }

    #[test]
    fn a_dropped_clause_does_not_reach_the_transcript() {
        // Gate G2b, end to end: whisper answered, but left a hole. The 8 s rule
        // does not help here -- the lane was not late, it was short.
        let fast = "THIS IS OUR FIRST MEETING SURPRISINGLY ENOUGH THIS IS OUR AGENDA";
        let mut s = dual();
        s.fast_final(&say(fast, 0.0, 0.3), t(3.3));
        let out = finals(&settle(&mut s, &say("This is our agenda.", 2.1, 0.3), 5.0));
        assert_eq!(out[0].lane, Lane::Fast);
        assert_eq!(out[0].reason, Some(FastReason::AccurateTruncated));
        assert_eq!(out[0].text, fast);
    }

    #[test]
    fn a_repetition_loop_does_not_reach_the_transcript() {
        // The mirror image (task 1.4): the accurate lane is not short, it is
        // four times too long. Words verbatim from the cropped-encoder run.
        let fast = "AND THEN WHEN YOU GO ON THE MENU YOU CAN SELECT";
        let loop_text = "and then when you go on the menu, you can select \
                         and then when you go on the menu, you can select \
                         and then when you go on the menu, you can select";
        let mut s = dual();
        s.fast_final(&say(fast, 0.0, 0.3), t(3.6));
        let out = finals(&settle(&mut s, &say(loop_text, 0.0, 0.1), 5.0));
        assert_eq!(out[0].lane, Lane::Fast);
        assert_eq!(out[0].reason, Some(FastReason::AccurateLooped));
        assert_eq!(out[0].text, fast);
    }

    #[test]
    fn an_empty_answer_never_blanks_a_line() {
        // The accurate lane committed past this line without producing a single
        // word inside it. Overwriting would put an empty line on screen and an
        // empty line in the transcript file §2.6 requires to be a record.
        let mut s = dual();
        s.fast_final(&say(FAST, 0.0, 0.35), t(1.4));
        let out = finals(&settle(&mut s, &say("later on", 2.0, 0.3), 5.0));
        assert_eq!(out[0].text, FAST);
        assert_eq!(out[0].reason, Some(FastReason::AccurateTruncated));
    }

    #[test]
    fn lines_reach_the_transcript_in_order_whatever_their_source() {
        let mut s = dual();
        s.fast_final(&say(FAST, 0.0, 0.35), t(1.4));
        s.fast_final(&say("AND THE TOPICS", 2.0, 0.4), t(3.2));
        // The accurate lane covers only the second line's audio.
        let out = finals(&settle(&mut s, &say("and the topics.", 2.0, 0.4), 5.0));
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].id, out[0].lane), (1, Lane::Fast));
        assert_eq!((out[1].id, out[1].lane), (2, Lane::Accurate));
    }

    #[test]
    fn accurate_words_for_a_promoted_line_do_not_leak_into_the_next_one() {
        let mut s = dual();
        s.fast_final(&say(FAST, 0.0, 0.35), t(1.4));
        s.fast_final(&say("AND THE TOPICS", 2.0, 0.4), t(3.2));
        // Line 1 gives up waiting.
        let out = finals(&s.tick(t(9.5)));
        assert_eq!(out[0].reason, Some(FastReason::AccurateTimeout));

        // ...and only then does whisper answer, for both spans at once.
        let hyp = [say(ACC, 0.0, 0.35), say("and the topics.", 2.0, 0.4)].concat();
        let out = finals(&settle(&mut s, &hyp, 10.0));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 2);
        assert_eq!(
            out[0].text, "and the topics.",
            "line 1's words were dropped"
        );
    }

    #[test]
    fn a_very_long_utterance_is_still_cut() {
        // The safety valve, not the normal path: sherpa's own endpoint rule
        // allows a 20 s utterance, and one subtitle line cannot be 20 s of
        // speech. A 26-word utterance is left alone; this one is not.
        let mut s = dual();
        let ordinary = (0..26)
            .map(|i| format!("W{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let ev = s.fast_final(&say(&ordinary, 0.0, 0.3), t(7.8));
        assert_eq!(partials(&ev).len(), 1);

        let long = (0..90)
            .map(|i| format!("W{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let ev = s.fast_final(&say(&long, 10.0, 0.2), t(28.0));
        let ids: Vec<u64> = partials(&ev).iter().map(|p| p.0).collect();
        assert_eq!(ids, [2, 3, 4]);
        assert_eq!(partials(&ev)[0].1.split_whitespace().count(), 40);
        assert_eq!(partials(&ev)[2].1.split_whitespace().count(), 10);
    }

    #[test]
    fn fast_only_mode_finalises_without_waiting() {
        let mut s = Stream::new(StreamConfig::default(), LaneMode::FastOnly);
        let out = finals(&s.fast_final(&say(FAST, 0.0, 0.35), t(1.4)));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].lane, Lane::Fast);
        assert_eq!(out[0].reason, Some(FastReason::AccurateDisabled));
        assert_eq!(out[0].span, (0.0, 1.4));
    }

    #[test]
    fn accurate_only_mode_cuts_its_own_lines() {
        // No fast lane to allocate line ids, so each utterance becomes one.
        let mut s = Stream::new(StreamConfig::default(), LaneMode::AccurateOnly);
        let long = (0..14)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let hyp = say(&long, 0.0, 0.3);
        let out = finals(&settle(&mut s, &hyp, 6.0));
        assert_eq!(out.len(), 1, "one utterance, one line");
        assert_eq!(out[0].text.split_whitespace().count(), 14);
        assert_eq!(out[0].lane, Lane::Accurate);
        assert_eq!(out[0].span, (0.0, 4.2));
    }

    #[test]
    fn the_frontier_and_the_prompt_follow_the_accurate_lane() {
        let mut s = dual();
        settle(&mut s, &say(ACC, 0.0, 0.35), 3.0);
        assert_eq!(s.frontier(), t(1.4));
        assert_eq!(s.prompt_tail(), ACC);
    }

    #[test]
    fn a_line_the_accurate_lane_answers_late_is_not_re_emitted() {
        // The 8 s rule fired, the line is on screen and in the transcript, and
        // only then does whisper answer for that audio. It is too late.
        let mut s = dual();
        s.fast_final(&say(FAST, 0.0, 0.35), t(1.4));
        assert_eq!(finals(&s.tick(t(9.5))).len(), 1);
        assert!(finals(&settle(&mut s, &say(ACC, 0.0, 0.35), 10.0)).is_empty());
    }
}
