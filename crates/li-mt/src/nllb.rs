//! The local backend: NLLB-200-distilled-600M, int8, on the CPU.
//!
//! MT deliberately does not touch the GPU. The prototype measured 0.2-0.3 s a
//! sentence on the CPU alone, and the accurate ASR lane needs every byte of
//! VRAM it can get.
//!
//! ## What this backend cannot do
//!
//! [`Translator::translate`] takes a `ctx` of preceding sentences. **NLLB
//! ignores it, on purpose.** Fed "I'm Sarah, the project manager. Hello
//! everybody." it returns 我是薩拉, 專案經理. -- the translation of the
//! *context*, with the sentence that was actually asked for dropped. It is a
//! sentence-level model with no notion of a document, and handing it two
//! sentences makes it choose one. `ctx` stays in the trait for the cloud and
//! LLM backends, where it is the whole point.
//!
//! ## Licence
//!
//! NLLB-200 is **CC-BY-NC**. Fine for a private build; it must be replaced
//! before anything is published or sold.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use ct2rs::sys::{ComputeType, Config, Device, TranslationOptions, Translator as Ct2};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{Translator, chunk, chunk::Marks, filler, zh::Zh};

/// NLLB's own name for English.
const SRC_LANG: &str = "eng_Latn";
/// NLLB's own name for **Simplified** Chinese, although the product writes
/// Traditional: [`Zh`] converts the characters and the vocabulary afterwards.
///
/// Asking for `zho_Hant` directly was worse on every count. Over 109 real
/// lines (72 from meetings, 37 read aloud, 2026-09-24), switching to
/// `zho_Hans` took the lines left hanging on a comma from 23 to 7, the
/// web-page boilerplate NLLB falls back on for short input (您的位置: 首頁,
/// 沒有任何問題) from 20 to 2, and chrF on the read clips from 16.9 to 19.1.
/// Its Traditional training data is the smaller and noisier half. The cost
/// is a longer decode: about 30% more time a line.
const TGT_LANG: &str = "zho_Hans";
const EOS: &str = "</s>";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NllbConfig {
    /// A CTranslate2 model directory: `model.bin` plus `tokenizer.json`.
    pub model_dir: PathBuf,
    /// 0 lets CTranslate2 choose. The ASR lanes are the ones under time
    /// pressure, so MT is not given the whole machine by default.
    pub threads: usize,
    /// 4, measured. Greedy (the prototype's setting) loses 1.5-2.4 chrF and drops
    /// clauses more often; 5 costs 70 ms more for nothing.
    pub beam_size: usize,
    /// Cut a line into clauses once it is longer than this (see [`chunk`]).
    /// 6 words: the point where the truncation rate stops falling.
    pub max_chunk_words: usize,
    /// Cut a stretch with no punctuation in it on word boundaries once it is
    /// longer than this. 0 leaves it whole, which is what every release up to
    /// the first width cut did -- and what left the fast lane, which punctuates nothing,
    /// handing NLLB 40 words at a time and getting one clause back.
    ///
    /// 6, measured, and the same number as `max_chunk_words` by measurement
    /// rather than by design: 5 is worse and 8 keeps a fifth of the content
    /// out of the translation.
    pub max_run_words: usize,
    /// Refuse to encode more than this. A runaway ASR line must not turn into a
    /// multi-second decode on the MT thread.
    pub max_input_tokens: usize,
    pub max_decoding_length: usize,
    /// Drop a chunk's final full stop before encoding (see
    /// [`chunk::trim_final_stop`]). **Off.** It reliably fixes one class of
    /// hallucination -- "Hello everybody." comes back as 您的位置: 首頁 and
    /// without the stop as 您好,所有人 -- but across the two scored clips chrF
    /// moved +1.9 and -1.5, and more lines ended mid-clause. A knob, not a
    /// default; the case it fixes is a short line padded into a hallucination.
    pub trim_final_stop: bool,
}

impl Default for NllbConfig {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            threads: 4,
            beam_size: 4,
            max_chunk_words: 6,
            max_run_words: 6,
            max_input_tokens: 200,
            max_decoding_length: 256,
            trim_final_stop: false,
        }
    }
}

/// The shared model cache, the same tree `li-asr` reads and the one task
/// downloader writes. `li-core` normally passes an explicit
/// directory; this is the standalone default.
pub fn default_model_dir() -> PathBuf {
    li_types::paths::model_cache().join("nllb-200-distilled-600m-ct2-int8")
}

pub struct LocalNllb {
    tok: Tokenizer,
    ct2: Ct2,
    zh: Zh,
    cfg: NllbConfig,
}

impl LocalNllb {
    pub fn open(cfg: &NllbConfig) -> Result<Self> {
        let dir: &Path = &cfg.model_dir;
        for f in ["model.bin", "tokenizer.json"] {
            if !dir.join(f).is_file() {
                bail!(
                    "MT model incomplete: {} is missing {f}\n\
                     Fetch nllb-200-distilled-600m-ct2-int8 or point \
                     `[mt] model_dir` at a CTranslate2 directory.",
                    dir.display()
                );
            }
        }

        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("loading {}/tokenizer.json", dir.display()))?;

        let ct2 = Ct2::new(
            dir,
            &Config {
                device: Device::CPU,
                compute_type: ComputeType::INT8,
                num_threads_per_replica: cfg.threads,
                ..Default::default()
            },
        )
        .with_context(|| format!("loading the CTranslate2 model in {}", dir.display()))?;

        tracing::info!(
            "MT: NLLB-200-distilled-600M int8 (CPU, {} threads, beam {})",
            cfg.threads,
            cfg.beam_size
        );
        Ok(Self {
            tok,
            ct2,
            zh: Zh::new()?,
            cfg: cfg.clone(),
        })
    }

    /// `["eng_Latn", "▁Hello", ..., "</s>"]`.
    ///
    /// Built here rather than by the tokenizer's own post-processor, which this
    /// model ships wrong: `tokenizer.json` carries the pre-fix NLLB template
    /// `A </s> <unk>` -- no source language at all, and a stray `<unk>`.
    /// Transformers patches it at run time from `src_lang`; nothing patches it
    /// for us.
    fn encode(&self, text: &str) -> Result<Vec<String>> {
        let enc = self.tok.encode(text, false).map_err(|e| anyhow!("{e}"))?;
        let body = enc.get_tokens();
        let n = body.len().min(self.cfg.max_input_tokens);
        let mut out = Vec::with_capacity(n + 2);
        out.push(SRC_LANG.to_string());
        out.extend_from_slice(&body[..n]);
        out.push(EOS.to_string());
        Ok(out)
    }

    fn decode(&self, hyp: &[String]) -> Result<String> {
        let body = match hyp.split_first() {
            Some((first, rest)) if first == TGT_LANG => rest,
            _ => hyp,
        };
        let ids: Vec<u32> = body
            .iter()
            .filter_map(|t| self.tok.token_to_id(t))
            .collect();
        Ok(self
            .tok
            .decode(&ids, true)
            .map_err(|e| anyhow!("{e}"))?
            .trim()
            .to_string())
    }

    /// One blocking pass. `li-core` runs this on a blocking thread.
    ///
    /// `marks` says where `src`'s punctuation came from; see [`chunk::Marks`].
    /// It is the caller's to answer because only the caller knows which lane
    /// produced the line, and now both of them punctuate.
    pub fn translate_blocking(&self, src: &str, marks: Marks) -> Result<String> {
        Ok(self.zh.finish(&self.pieces(src, marks)?))
    }

    /// One string per clause, before OpenCC and before punctuation: what the
    /// model produced, or for a clause of nothing but interjections, what
    /// [`filler::render`] wrote in its place. Exposed so a test can compare it
    /// against the prototype's Python implementation without the
    /// post-processing in the way.
    pub fn pieces(&self, src: &str, marks: Marks) -> Result<Vec<String>> {
        let pieces = chunk::split(src, self.cfg.max_chunk_words, self.cfg.max_run_words, marks);
        // "Mm-hmm." alone comes back from the model as 沒有任何問題; see
        // `filler`. Those clauses never reach it, the rest go as one batch.
        let mut out: Vec<Option<String>> = pieces.iter().map(|p| filler::render(p)).collect();
        let rest: Vec<&str> = pieces
            .iter()
            .zip(&out)
            .filter(|(_, done)| done.is_none())
            .map(|(p, _)| *p)
            .collect();
        let mut translated = self.model(&rest)?.into_iter();
        for slot in out.iter_mut().filter(|s| s.is_none()) {
            *slot = translated.next();
        }
        Ok(out.into_iter().flatten().collect())
    }

    /// The pieces through CTranslate2 in one batch, one output per piece.
    fn model(&self, pieces: &[&str]) -> Result<Vec<String>> {
        if pieces.is_empty() {
            return Ok(Vec::new());
        }
        let source = pieces
            .iter()
            .map(|p| {
                self.encode(if self.cfg.trim_final_stop {
                    chunk::trim_final_stop(p)
                } else {
                    p
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let prefix = vec![vec![TGT_LANG]; source.len()];

        let opts = TranslationOptions {
            beam_size: self.cfg.beam_size,
            max_decoding_length: self.cfg.max_decoding_length,
            ..Default::default()
        };
        let results = self
            .ct2
            .translate_batch_with_target_prefix(&source, &prefix, &opts, None)
            .context("CTranslate2 translate_batch")?;

        results
            .iter()
            .map(|r| self.decode(r.output().map(Vec::as_slice).unwrap_or_default()))
            .collect()
    }

    /// The token sequence handed to CTranslate2 for `text`, for tests.
    pub fn source_tokens(&self, text: &str) -> Result<Vec<String>> {
        self.encode(text)
    }
}

#[async_trait]
impl Translator for LocalNllb {
    async fn translate(&self, src: &str, _ctx: &[String]) -> Result<String> {
        // The trait has no lane in it and its one caller is a test; the
        // conservative answer is today's rule.
        self.translate_blocking(src, Marks::Heard)
    }

    /// The model writes Simplified Chinese, so this backend converts its own
    /// output and answers `true`. See [`Zh`].
    fn target_is_zh_tw(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_model_directory_is_the_shared_model_cache() {
        let d = default_model_dir();
        assert!(d.ends_with("nllb-200-distilled-600m-ct2-int8"));
        assert!(
            d.to_string_lossy()
                .contains(".cache/liveinterpreter/models")
        );
    }

    #[test]
    fn a_missing_model_says_which_file_and_where() {
        let cfg = NllbConfig {
            model_dir: "/nonexistent/mt".into(),
            ..Default::default()
        };
        let err = LocalNllb::open(&cfg)
            .err()
            .expect("a missing model is an error")
            .to_string();
        assert!(err.contains("model.bin"), "{err}");
        assert!(err.contains("/nonexistent/mt"), "{err}");
    }

    #[test]
    fn the_config_round_trips_through_toml() {
        let cfg = NllbConfig::default();
        let back: NllbConfig = toml::from_str(&toml::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(cfg, back);
    }
}
