//! The committed transcript, and what the accurate lane is told about it.
//!
//! ## What used to be here, and why it is not
//!
//! The first design specified **LocalAgreement-2**: re-transcribe a rolling buffer
//! every tick and commit a word once two consecutive hypotheses agree on it
//! (ufal/whisper_streaming).
//! It was implemented, measured, and removed.
//!
//! The measurement, all on `ami_meeting2`, `small.en` q5_1 on Vulkan, against a
//! 15.2% content WER for the same model transcribing the same audio offline:
//!
//! | accurate-lane policy | content WER | drop rate |
//! |---|---|---|
//! | rolling buffer trimmed to the commit frontier each tick | 27.8% | 0.0% |
//! | ...trimmed only at silences, sliding 6 s window | 31.7% | 8.7% |
//! | ...growing to 12 s then starting over | 40.0% | 15.7% |
//! | **one pass over one whole utterance** | **17.8%** | **0.0%** |
//!
//! Two things go wrong at once. Whisper is much worse at a window that starts
//! mid-clause -- it condenses six seconds of speech into six words and repeats
//! itself -- so every re-transcription of a trimmed buffer is a worse
//! hypothesis than a transcription of the whole utterance. And because it
//! revises itself freely between passes, two consecutive hypotheses often never
//! agree, so nothing commits until something forces it, and what gets forced
//! out is the worst hypothesis rather than the best.
//!
//! LocalAgreement-2 exists to get *early* text out of a lane that is still
//! listening. In this architecture the fast lane already did that, 1.1 s
//! earlier, and the accurate lane's only job is to be right. So it waits for an
//! endpoint -- the fast lane's own, or the VAD's -- and transcribes that
//! utterance once. It is also three times cheaper, because each second of audio
//! is now transcribed once instead of about eight times: measured engine time
//! per audio second fell from 0.41 to 0.15.

use std::time::Duration;

use li_types::Word;

use crate::text;

/// Everything the accurate lane has finally said, in order.
#[derive(Debug, Default)]
pub struct Committed {
    words: Vec<Word>,
    /// End of the last committed word, on the audio timeline.
    frontier: Duration,
}

impl Committed {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn frontier(&self) -> Duration {
        self.frontier
    }

    /// Take a finished transcription of one whole utterance.
    pub fn adopt(&mut self, words: &[Word]) -> Vec<Word> {
        if let Some(last) = words.last() {
            self.frontier = last.end.max(self.frontier);
        }
        self.words.extend(words.iter().cloned());
        words.to_vec()
    }

    /// The tail of the committed text, for whisper's `initial_prompt`.
    ///
    /// The prototype measured that carrying it improves recognition at the start of a
    /// buffer, which is every pass for a lane that transcribes one utterance at
    /// a time.
    pub fn prompt_tail(&self, max_chars: usize) -> String {
        let all = text::join(&self.words);
        match all.char_indices().nth_back(max_chars.saturating_sub(1)) {
            Some((i, _)) => all[i..].to_owned(),
            None => all,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hyp(spec: &[(&str, f64, f64)]) -> Vec<Word> {
        spec.iter()
            .map(|(t, s, e)| Word {
                text: (*t).into(),
                start: Duration::from_secs_f64(*s),
                end: Duration::from_secs_f64(*e),
            })
            .collect()
    }

    #[test]
    fn the_frontier_follows_the_last_word_taken() {
        let mut c = Committed::new();
        c.adopt(&hyp(&[("ask", 0.0, 0.4), ("not", 0.4, 0.7)]));
        assert_eq!(c.frontier(), Duration::from_secs_f64(0.7));
        c.adopt(&hyp(&[("what", 1.2, 1.6)]));
        assert_eq!(c.frontier(), Duration::from_secs_f64(1.6));
    }

    #[test]
    fn an_empty_utterance_does_not_move_the_frontier_backwards() {
        let mut c = Committed::new();
        c.adopt(&hyp(&[("ask", 0.0, 0.4)]));
        c.adopt(&[]);
        assert_eq!(c.frontier(), Duration::from_secs_f64(0.4));
    }

    #[test]
    fn the_prompt_tail_is_the_end_of_the_committed_text() {
        let mut c = Committed::new();
        c.adopt(&hyp(&[
            ("ask", 0.0, 0.4),
            ("not", 0.4, 0.7),
            ("what", 0.7, 1.0),
        ]));
        assert_eq!(c.prompt_tail(200), "ask not what");
        assert_eq!(c.prompt_tail(4), "what");
    }
}
