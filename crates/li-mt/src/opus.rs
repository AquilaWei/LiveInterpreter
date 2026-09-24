//! The second opinion: Opus-MT en-zh, asked only when NLLB could not write a
//! character.
//!
//! NLLB-200's vocabulary has holes in it. 饋 (反饋, "feedback"), 碩 (碩士,
//! "master's degree"), 鋰 ("lithium") and a few more are not in it, so when the
//! translation needs one the model emits `<unk>`. Decoded the usual way that
//! token simply vanishes, and the line reads wrong rather than incomplete:
//! "your feedback" came back 你的反, "an LLM" 法學士, and once 你的士 -- which
//! OpenCC then made 你計程車 ("you taxi"). Over 109 real lines it happened 21
//! times (2026-09-24).
//!
//! Opus-MT is a different model with a different vocabulary, small enough (77M
//! parameters, 80 MB) to keep loaded for the minority of pieces that need it,
//! and quick (about 50 ms a piece on the CPU). Its own translations are
//! literal -- "Great, great." is 偉大的，偉大的 -- so it is not the main
//! translator, but for writing the word NLLB could not it only has to be
//! sound.
//!
//! ## How it is asked
//!
//! * **Simplified, not Traditional.** Its `>>cmn_Hant<<` output stops after
//!   the first sentence of a line far more often (12 of 34 meeting lines
//!   against 2 with `>>cmn_Hans<<`); [`crate::Zh`] converts afterwards, as it
//!   does for NLLB.
//! * **One sentence at a time.** It is a sentence-level model like NLLB, and
//!   handed two sentences it translates one. The pieces it gets are NLLB's
//!   clauses, which [`crate::chunk::split`] has already cut, so this is
//!   usually one call.
//!
//! ## When its answer is not used
//!
//! See [`usable`]. The caller then keeps NLLB's translation with
//! [`crate::nllb::MISSING`] where the character was.
//!
//! ## Licence
//!
//! Helsinki-NLP/opus-mt-en-zh is **CC-BY-4.0**: commercial use is allowed,
//! with attribution (see NOTICE).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ct2rs::sys::{ComputeType, Config, Device, TranslationOptions, Translator as Ct2};
use tokenizers::Tokenizer;

use crate::spm;

/// Opus-MT's multi-target models pick the script with a token at the start of
/// the source.
const TARGET: &str = ">>cmn_Hans<<";
const EOS: &str = "</s>";
const UNK: &str = "<unk>";

/// The shared model cache, beside NLLB.
pub fn default_model_dir() -> PathBuf {
    li_types::paths::model_cache().join("opus-mt-en-zh-ct2-int8")
}

pub struct LocalOpus {
    ct2: Ct2,
    /// `source.spm`. The target side needs no tokenizer: the model returns
    /// pieces, and turning pieces into text is [`join_pieces`].
    source: Tokenizer,
}

impl LocalOpus {
    /// Fails when a file is missing or unreadable. Whether a missing model is
    /// an error is the caller's call; see `NllbConfig::fallback_model_dir`.
    pub fn open(dir: &Path, threads: usize) -> Result<Self> {
        for f in ["model.bin", "source.spm"] {
            if !dir.join(f).is_file() {
                bail!(
                    "fallback MT model incomplete: {} is missing {f}",
                    dir.display()
                );
            }
        }
        let ct2 = Ct2::new(
            dir,
            &Config {
                device: Device::CPU,
                compute_type: ComputeType::INT8,
                num_threads_per_replica: threads,
                ..Default::default()
            },
        )
        .with_context(|| format!("loading the CTranslate2 model in {}", dir.display()))?;
        tracing::info!("MT fallback: Opus-MT en-zh int8 (CPU, {threads} threads)");
        Ok(Self {
            ct2,
            source: spm::unigram(&dir.join("source.spm"))?,
        })
    }

    /// `piece` in Simplified Chinese, or `None` when the answer is not one to
    /// put on screen (see [`usable`]).
    pub fn translate(&self, piece: &str) -> Result<Option<String>> {
        let mut source = vec![TARGET.to_string()];
        source.extend_from_slice(self.source_pieces(piece)?.as_slice());
        source.push(EOS.to_string());
        let opts = TranslationOptions {
            beam_size: 4,
            max_decoding_length: 256,
            ..Default::default()
        };
        let results = self
            .ct2
            .translate_batch(&[source], &opts, None)
            .context("CTranslate2 translate_batch (fallback)")?;
        let hyp = results
            .first()
            .and_then(|r| r.output())
            .cloned()
            .unwrap_or_default();
        if hyp.iter().any(|t| t == UNK) {
            return Ok(None);
        }
        let text = close_up(&join_pieces(&hyp));
        Ok(usable(&text).then_some(text))
    }

    /// `text` as the pieces the model reads, without the target token and the
    /// end of sentence. Public for the test that checks them against
    /// SentencePiece's own.
    pub fn source_pieces(&self, text: &str) -> Result<Vec<String>> {
        Ok(self
            .source
            .encode(text, false)
            .map_err(|e| anyhow!("{e}"))?
            .get_tokens()
            .to_vec())
    }
}

/// SentencePiece's decoding, which is all there is to it for a Unigram model:
/// the pieces joined, with the word-boundary mark `▁` read as a space.
fn join_pieces(pieces: &[String]) -> String {
    pieces.concat().replace('▁', " ")
}

/// Whether a fallback translation can stand in for NLLB's.
///
/// Its own `<unk>` is checked before decoding. After it, two things mean the
/// model had nothing sensible to say, both seen on real input: an empty answer,
/// and a character from Unicode's private use area -- "And then, uh..." came
/// back as 礛 followed by two of them, which render as nothing or as a box and
/// are never Chinese.
pub fn usable(text: &str) -> bool {
    !text.trim().is_empty() && !text.chars().any(|c| ('\u{E000}'..='\u{F8FF}').contains(&c))
}

/// Opus-MT's target vocabulary starts a piece with a word-boundary mark even
/// inside Chinese, so a decoded line reads "我不断得到 很多要求". Chinese has
/// no spaces between words; a space survives only between two Latin letters
/// or digits, where it is part of the text: "3 GHz" keeps its space, and
/// "LLM 越来越大" loses it.
fn close_up(s: &str) -> String {
    let chars: Vec<char> = s.trim().chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            let before = out.chars().next_back();
            let after = chars[i + 1..].iter().find(|c| !c.is_whitespace());
            let latin = |c: Option<&char>| c.is_some_and(|c| c.is_ascii_alphanumeric());
            if latin(before.as_ref()) && latin(after) && !out.ends_with(' ') {
                out.push(' ');
            }
            continue;
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ordinary_translation_is_usable() {
        assert!(usable("我真的很想得到你的反馈。"));
    }

    #[test]
    fn an_empty_translation_is_not_usable() {
        assert!(!usable("  "));
    }

    #[test]
    fn a_private_use_character_makes_a_translation_unusable() {
        assert!(!usable("礛\u{e09e}\u{e4d1}"));
    }

    #[test]
    fn the_spaces_between_chinese_words_are_closed_up() {
        assert_eq!(close_up("我不断得到 很多要求"), "我不断得到很多要求");
    }

    #[test]
    fn a_space_between_two_latin_words_is_kept() {
        assert_eq!(close_up("频率是 3 GHz"), "频率是3 GHz");
    }

    #[test]
    fn pieces_are_joined_with_the_word_mark_as_a_space() {
        let pieces = ["▁我", "不断", "得到", "▁很多"].map(String::from);
        assert_eq!(join_pieces(&pieces), " 我不断得到 很多");
    }

    #[test]
    fn a_missing_model_says_which_file_and_where() {
        let err = LocalOpus::open(Path::new("/nonexistent/opus"), 1)
            .err()
            .expect("a missing model is an error")
            .to_string();
        assert!(err.contains("model.bin"), "{err}");
        assert!(err.contains("/nonexistent/opus"), "{err}");
    }
}
