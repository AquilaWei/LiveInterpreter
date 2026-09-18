//! Punctuation and casing restored from the words, not from the silence.
//!
//! The fast lane emits neither. Until now the only thing that knew where a
//! sentence ended was sherpa's endpoint detector, which knows it by *listening*
//! — and a speaker going fast enough leaves no silence to listen to. Measurement
//! tried both cheap ways of listening harder and closed them: the VAD sees a
//! strict subset of what sherpa's own counter sees, and dropping
//! `endpoint_silence_s` to 0.45 costs `ami_meeting.wav` 7 points of content WER.
//!
//! The information is there anyway. In the user's own five sessions, **77
//! sentence-final marks fell mid-line in the accurate lane's text** across 115
//! lines — that is whisper, a model that reads words rather than silence,
//! finding 77 boundaries the endpoint detector missed on the same audio. This
//! module is the cheap version of that faculty, running on the fast lane's text
//! where it is early enough to be worth something.
//!
//! ## What it is
//!
//! sherpa-onnx's *online* punctuation API, which is already in the `.so` we
//! link and already in the generated bindings — a CNN-BiLSTM from the
//! Edge-Punct-Casing work, 7.5 MB at int8, English only. `AddPunct` is a pure
//! text-to-text call with no state carried between invocations.
//!
//! ## Two things the C++ will not tell you
//!
//! **It must be fed lower case.** `DecodeSentences` in
//! `online-punctuation-cnn-bilstm-impl.h` only ever calls `toupper`; there is
//! no path that lowers a character. The fast lane's output is entirely upper
//! case, so handing it over untouched returns it unchanged and uppercase, with
//! punctuation in the right places and no casing signal at all. [`restore`]
//! lowercases first.
//!
//! **It may not preserve the word count.** Nothing in the C API promises it,
//! and everything downstream assumes it: [`restore_words`] keeps each [`Word`]'s
//! `start` and `end` and swaps only its `text`, which is what lets
//! `li_stream::Stream::fast_final` chunk the result and what lets the two lanes
//! stay aligned on the audio timeline. So the count is checked rather than
//! trusted, and a mismatch returns the input untouched.

// Same rules as `sherpa.rs`: the pointer is created and destroyed by this type,
// never handed out, and only touched through `&mut self`.
#![allow(unsafe_code)]

use std::{
    ffi::{CStr, CString},
    mem,
    path::{Path, PathBuf},
    ptr,
};

use anyhow::{Result, anyhow, bail};
use li_types::Word;
use sherpa_rs_sys as sys;

/// The release directory's own names. `model.int8.onnx` is preferred over the
/// 29 MB float one: this runs on the pipeline thread next to the fast lane, and
/// the float model buys nothing a subtitle can see.
const MODEL_CANDIDATES: [&str; 2] = ["model.int8.onnx", "model.onnx"];
const VOCAB: &str = "bpe.vocab";

pub struct OnlinePunct {
    punct: *const sys::SherpaOnnxOnlinePunctuation,
    model: String,
    /// Word-count mismatches seen so far. Logged once, then counted silently —
    /// a model that has started disagreeing will disagree on every line, and a
    /// warning per line would be the loudest thing in the journal.
    mismatches: u64,
}

// Owned by this value, reachable only through `&mut self`, freed in `Drop`.
// (`Sync` is deliberately not claimed; see the same note in `sherpa.rs`.)
unsafe impl Send for OnlinePunct {}

impl OnlinePunct {
    /// `dir` is a `sherpa-onnx-online-punct-*` release directory.
    pub fn open(dir: &Path, threads: usize) -> Result<Self> {
        if !dir.is_dir() {
            bail!(
                "punctuation model directory not found: {}\n\
                 Fetch sherpa-onnx-online-punct-en-2024-08-06 or turn \
                 `[asr.fast] punctuation` off.",
                dir.display()
            );
        }
        let model = MODEL_CANDIDATES
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.is_file())
            .ok_or_else(|| {
                anyhow!(
                    "{} has no {}\n\
                     It should be a sherpa-onnx online punctuation release.",
                    dir.display(),
                    MODEL_CANDIDATES.join(" or ")
                )
            })?;
        let vocab = dir.join(VOCAB);
        if !vocab.is_file() {
            bail!("{} is missing {VOCAB}", dir.display());
        }

        // sherpa copies these, so they only have to outlive the create call.
        let model_c = cstring(&model)?;
        let vocab_c = cstring(&vocab)?;
        let provider = CString::new("cpu")?;

        let punct = unsafe {
            let mut cfg: sys::SherpaOnnxOnlinePunctuationConfig = mem::zeroed();
            cfg.model.cnn_bilstm = model_c.as_ptr();
            cfg.model.bpe_vocab = vocab_c.as_ptr();
            cfg.model.num_threads = threads.max(1) as i32;
            cfg.model.provider = provider.as_ptr();
            sys::SherpaOnnxCreateOnlinePunctuation(&cfg)
        };
        if punct.is_null() {
            bail!(
                "sherpa-onnx refused the punctuation model in {} -- check that \
                 the .onnx and {VOCAB} come from the same release",
                dir.display()
            );
        }

        let name = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.display().to_string());
        tracing::info!("punctuation: {name} (CPU, {threads} threads)");
        Ok(Self {
            punct,
            model: name,
            mismatches: 0,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// How many lines came back with a different word count than they went in
    /// with. Anything but 0 means [`restore_words`] has been handing text back
    /// unchanged, and the caller should say so rather than report a win.
    pub fn mismatches(&self) -> u64 {
        self.mismatches
    }

    /// Punctuate and case one line. Empty in, empty out.
    pub fn restore(&mut self, text: &str) -> Result<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(String::new());
        }
        // Lower case, or the model has nothing to say about casing and returns
        // the shouting it was given. See the module docs.
        let lowered = CString::new(trimmed.to_lowercase())?;
        let out = unsafe {
            let p = sys::SherpaOnnxOnlinePunctuationAddPunct(self.punct, lowered.as_ptr());
            if p.is_null() {
                bail!("sherpa-onnx returned no punctuated text");
            }
            let s = CStr::from_ptr(p).to_string_lossy().into_owned();
            // Its own allocator, not `free`: the C API hands out `new[]`.
            sys::SherpaOnnxOnlinePunctuationFreeText(p);
            s
        };
        Ok(out.trim().to_string())
    }

    /// The same, over timed words.
    ///
    /// Times are carried through untouched — this only ever rewrites `text`.
    /// If the model changes how many words there are, the input is returned
    /// unchanged: a `Word` list whose text no longer lines up with its
    /// timestamps would put line boundaries in the wrong place, which is worse
    /// than having no punctuation at all.
    pub fn restore_words(&mut self, words: &[Word]) -> Result<Vec<Word>> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let joined = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let punctuated = self.restore(&joined)?;
        let pieces: Vec<&str> = punctuated.split_whitespace().collect();
        if pieces.len() != words.len() || !same_words(&pieces, words) {
            self.mismatches += 1;
            if self.mismatches == 1 {
                tracing::warn!(
                    "punctuation changed the words, not just the marks: {} in, {} out \
                     -- leaving the fast lane's text alone from here",
                    words.len(),
                    pieces.len()
                );
            }
            return Ok(words.to_vec());
        }
        Ok(words
            .iter()
            .zip(pieces)
            .map(|(w, text)| Word {
                text: text.to_owned(),
                start: w.start,
                end: w.end,
            })
            .collect())
    }
}

impl Drop for OnlinePunct {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOnlinePunctuation(self.punct) };
        self.punct = ptr::null();
    }
}

/// Are these the same words, ignoring the marks and the casing this module
/// exists to add?
fn same_words(pieces: &[&str], words: &[Word]) -> bool {
    pieces
        .iter()
        .zip(words)
        .all(|(p, w)| bare(p) == bare(&w.text))
}

fn bare(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

fn cstring(p: &Path) -> Result<CString> {
    let s = p
        .to_str()
        .ok_or_else(|| anyhow!("model path is not UTF-8: {}", p.display()))?;
    CString::new(s).map_err(Into::into)
}

/// The shared model cache's name for the released model, for callers that have
/// no config file to read.
pub fn default_model_dir() -> PathBuf {
    li_types::paths::model_cache().join("sherpa-onnx-online-punct-en-2024-08-06")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn word(text: &str, s: u64) -> Word {
        Word {
            text: text.into(),
            start: Duration::from_millis(s),
            end: Duration::from_millis(s + 200),
        }
    }

    #[test]
    fn a_missing_directory_says_where_and_how_to_turn_it_off() {
        let err = OnlinePunct::open(Path::new("/nonexistent/punct"), 1)
            .err()
            .expect("a missing model is an error")
            .to_string();
        assert!(err.contains("/nonexistent/punct"), "{err}");
        assert!(err.contains("punctuation"), "{err}");
    }

    #[test]
    fn the_word_comparison_ignores_exactly_the_marks_and_the_casing() {
        let words = [word("NOTHING", 0), word("CAN", 200), word("SURPRISE", 400)];
        assert!(same_words(&["Nothing", "can", "surprise."], &words));
        assert!(same_words(&["\"Nothing", "can", "surprise!\""], &words));
        // ...and nothing else. A dropped or invented word has to be caught.
        assert!(!same_words(&["Nothing", "can", "astonish."], &words));
    }

    #[test]
    fn an_apostrophe_is_part_of_the_word_and_not_a_mark() {
        // "I'M" and "I'm," are the same word; "im" is not, and treating the
        // apostrophe as punctuation would make all three compare equal.
        let words = [word("I'M", 0)];
        assert!(same_words(&["I'm,"], &words));
        assert!(!same_words(&["im"], &words));
    }
}
