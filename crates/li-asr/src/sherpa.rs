//! Fast lane: sherpa-onnx streaming Zipformer (PLAN §8.2).
//!
//! ## Why this is FFI and not `sherpa-rs`
//!
//! `sherpa-rs`, the safe wrapper, binds sherpa-onnx's **offline** recognizer
//! for ASR. Its `transducer` and `zipformer` modules both call
//! `SherpaOnnxCreateOfflineRecognizer`; the online API appears in the crate
//! only for keyword spotting and speaker id. An offline recognizer re-runs the
//! encoder over whatever you hand it, which is the accurate lane's shape, not
//! this one's — using it here would give up the single property the fast lane
//! exists for: latency that does not grow with the buffer (§12.1).
//!
//! So this lane talks to `sherpa-rs-sys` directly. The surface is eight
//! functions and it is the same one the Phase 0 PoC drove through Python.
//!
//! ## Segmentation is the backend's
//!
//! sherpa-onnx has its own endpoint detector, and `rule2` (trailing silence
//! after a decoded token) is deliberately set to the same 0.6 s as
//! `li_stream::StreamConfig::pause_flush_s`. The alternative — re-deriving
//! sentence breaks downstream — has to work from text with no punctuation and
//! no casing, which is exactly the information the endpoint detector has and
//! the text does not.

// The C API is the only way to reach the streaming recognizer (see above).
// Every raw pointer here is created and destroyed by this type, is never
// handed out, and is only touched through `&mut self`.
#![allow(unsafe_code)]

use std::{
    ffi::{CStr, CString},
    mem,
    path::{Path, PathBuf},
    ptr,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use li_types::AsrEvent;
use serde::Deserialize;
use sherpa_rs_sys as sys;

use crate::{
    AsrCaps, AsrEngine, LaneSpec,
    device::{Accel, BackendInfo, Selection},
    words,
};

const SAMPLE_RATE: i32 = 16_000;
/// Zipformer's fbank dimension. Not a free parameter — it has to match the model.
const FEATURE_DIM: i32 = 80;

/// Trailing silence that ends a sentence when nothing has been decoded yet.
const RULE1_SILENCE: f32 = 2.4;

/// Nominal duration of the final piece of a segment, which has no successor to
/// bound it. Same value as the Phase 0 PoC.
const LAST_PIECE: Duration = Duration::from_millis(120);

/// Silence pushed through the encoder by [`AsrEngine::finalize`] so the tail of
/// the last word is actually decoded rather than left in the feature buffer.
const FLUSH_SILENCE: Duration = Duration::from_millis(500);

pub struct SherpaFast {
    recognizer: *const sys::SherpaOnnxOnlineRecognizer,
    stream: *const sys::SherpaOnnxOnlineStream,
    info: BackendInfo,
    seg_id: u64,
    last_partial: String,
    /// Samples handed to the stream so far. sherpa's timestamps are on this
    /// clock, which is not the audio timeline whenever `li-audio` has dropped a
    /// frame — hence `drift`.
    fed: u64,
    drift: f64,
}

// The recognizer and stream are owned by this value, reachable only through
// `&mut self`, and freed in `Drop`. sherpa-onnx supports using them from
// another thread; it does not support two threads at once, which `&mut` rules
// out. (`Sync` is deliberately not claimed.)
unsafe impl Send for SherpaFast {}

impl SherpaFast {
    pub fn open(spec: &LaneSpec) -> Result<Self> {
        let dir = &spec.model;
        if !dir.is_dir() {
            bail!(
                "fast-lane model directory not found: {}\n\
                 Download a streaming Zipformer (PLAN §15) or point `[asr.fast] model` at one.",
                dir.display()
            );
        }
        let files = ModelFiles::find(dir)?;
        let threads = spec.threads.max(1) as i32;

        // The recognizer copies these strings, so the CStrings only have to
        // outlive the create call.
        let encoder = cstring(&files.encoder)?;
        let decoder = cstring(&files.decoder)?;
        let joiner = cstring(&files.joiner)?;
        let tokens = cstring(&files.tokens)?;
        let provider = CString::new("cpu")?;
        let decoding = CString::new("greedy_search")?;

        let recognizer = unsafe {
            let mut cfg: sys::SherpaOnnxOnlineRecognizerConfig = mem::zeroed();
            cfg.feat_config.sample_rate = SAMPLE_RATE;
            cfg.feat_config.feature_dim = FEATURE_DIM;
            cfg.model_config.transducer.encoder = encoder.as_ptr();
            cfg.model_config.transducer.decoder = decoder.as_ptr();
            cfg.model_config.transducer.joiner = joiner.as_ptr();
            cfg.model_config.tokens = tokens.as_ptr();
            cfg.model_config.num_threads = threads;
            cfg.model_config.provider = provider.as_ptr();
            cfg.decoding_method = decoding.as_ptr();
            cfg.enable_endpoint = 1;
            cfg.rule1_min_trailing_silence = RULE1_SILENCE;
            // The sentence-break rule, and the largest single term in G1: no
            // line can be closed sooner than this after the speaker stops.
            // `LaneSpec::endpoint_silence_s` is the knob.
            cfg.rule2_min_trailing_silence = spec.endpoint_silence_s;
            cfg.rule3_min_utterance_length = spec.max_utterance_s;
            sys::SherpaOnnxCreateOnlineRecognizer(&cfg)
        };
        if recognizer.is_null() {
            bail!(
                "sherpa-onnx refused the model in {} -- check that encoder/decoder/joiner \
                 come from the same release",
                dir.display()
            );
        }
        let stream = unsafe { sys::SherpaOnnxCreateOnlineStream(recognizer) };
        if stream.is_null() {
            unsafe { sys::SherpaOnnxDestroyOnlineRecognizer(recognizer) };
            bail!("sherpa-onnx could not create a stream");
        }

        Ok(Self {
            recognizer,
            stream,
            info: BackendInfo {
                engine: "sherpa-onnx",
                model: dir
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| dir.display().to_string()),
                // The fast lane is CPU by decision, not by fallback: it costs
                // ~0.3 GB and one core, and leaving the GPU entirely to the
                // accurate lane is the point (§8.1).
                selection: Selection {
                    accel: Accel::Cpu,
                    gpu_index: 0,
                    device: format!("CPU ({threads} threads)"),
                    note: None,
                },
            },
            seg_id: 0,
            last_partial: String::new(),
            fed: 0,
            drift: 0.0,
        })
    }

    fn decode_ready(&mut self) {
        unsafe {
            while sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) != 0 {
                sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
            }
        }
    }

    fn partial_text(&mut self) -> String {
        unsafe {
            let r = sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
            if r.is_null() {
                return String::new();
            }
            let text = if (*r).text.is_null() {
                String::new()
            } else {
                CStr::from_ptr((*r).text).to_string_lossy().into_owned()
            };
            sys::SherpaOnnxDestroyOnlineRecognizerResult(r);
            // Defensive: some builds leave the word marker in `text`, and a
            // marker that stands alone as its own token then leaves a double
            // space behind.
            text.replace('\u{2581}', " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    /// Read the current segment as words and clear it.
    fn take_segment(&mut self) -> Result<Option<AsrEvent>> {
        let json = unsafe {
            let p = sys::SherpaOnnxGetOnlineStreamResultAsJson(self.recognizer, self.stream);
            if p.is_null() {
                return Ok(None);
            }
            let s = CStr::from_ptr(p).to_string_lossy().into_owned();
            sys::SherpaOnnxDestroyOnlineStreamResultJson(p);
            s
        };
        unsafe { sys::SherpaOnnxOnlineStreamReset(self.recognizer, self.stream) };
        self.last_partial.clear();

        let raw: SherpaResult =
            serde_json::from_str(&json).context("parsing the sherpa-onnx result")?;
        if raw.tokens.is_empty() {
            // An endpoint with nothing decoded: rule1 fired on silence. There
            // is no sentence here, only a boundary.
            return Ok(None);
        }
        if raw.timestamps.len() != raw.tokens.len() {
            bail!(
                "sherpa-onnx returned {} tokens but {} timestamps",
                raw.tokens.len(),
                raw.timestamps.len()
            );
        }

        let base = f64::from(raw.start_time) + self.drift;
        let pieces: Vec<(String, Duration)> = raw
            .tokens
            .iter()
            .zip(&raw.timestamps)
            .map(|(tok, ts)| (tok.clone(), secs(base + f64::from(*ts))))
            .collect();
        let t_end = pieces
            .last()
            .map(|(_, t)| *t + LAST_PIECE)
            .unwrap_or_default();
        let words = words::merge(&pieces, t_end);
        let Some(first) = words.first() else {
            return Ok(None);
        };
        let event = AsrEvent::Final {
            seg_id: self.seg_id,
            t_start: first.start,
            t_end,
            words,
        };
        self.seg_id += 1;
        Ok(Some(event))
    }

    fn accept(&mut self, pcm: &[f32]) {
        if pcm.is_empty() {
            return;
        }
        unsafe {
            sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                SAMPLE_RATE,
                pcm.as_ptr(),
                pcm.len() as i32,
            );
        }
        self.fed += pcm.len() as u64;
    }
}

#[async_trait]
impl AsrEngine for SherpaFast {
    fn capabilities(&self) -> AsrCaps {
        AsrCaps {
            native_streaming: true,
            punctuated: false,
        }
    }

    fn backend(&self) -> &BackendInfo {
        &self.info
    }

    async fn feed(&mut self, pcm: &[f32], t_origin: Duration) -> Result<Vec<AsrEvent>> {
        // sherpa counts the audio it was given; the timeline counts the audio
        // that happened. They differ by whatever `li-audio` dropped.
        self.drift = t_origin.as_secs_f64() - self.fed as f64 / f64::from(SAMPLE_RATE);
        self.accept(pcm);
        self.decode_ready();

        let mut out = Vec::new();
        let endpoint =
            unsafe { sys::SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) != 0 };
        if endpoint {
            if let Some(ev) = self.take_segment()? {
                out.push(ev);
            }
            return Ok(out);
        }
        let text = self.partial_text();
        if !text.is_empty() && text != self.last_partial {
            self.last_partial.clone_from(&text);
            out.push(AsrEvent::Partial {
                seg_id: self.seg_id,
                text,
                // Timings for a partial would cost a JSON round trip 30x a
                // second for a line that is about to be overwritten. The fast
                // lane's partials are screen-only (§8.2); nothing downstream
                // aligns on them.
                words: Vec::new(),
            });
        }
        Ok(out)
    }

    async fn finalize(&mut self) -> Result<Option<AsrEvent>> {
        // Push silence rather than `InputFinished`: the tail of the last word
        // is still in the feature buffer, and `InputFinished` would also close
        // the stream for good, which makes this callable exactly once.
        let pad = vec![0.0f32; (FLUSH_SILENCE.as_secs_f64() * f64::from(SAMPLE_RATE)) as usize];
        self.accept(&pad);
        self.decode_ready();
        self.take_segment()
    }
}

impl Drop for SherpaFast {
    fn drop(&mut self) {
        unsafe {
            sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
        self.stream = ptr::null();
        self.recognizer = ptr::null();
    }
}

#[derive(Debug, Deserialize)]
struct SherpaResult {
    tokens: Vec<String>,
    timestamps: Vec<f32>,
    /// Where this segment starts on the stream clock. Reset moves it forward,
    /// so segment-relative timestamps stay usable across sentence breaks.
    start_time: f32,
}

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s.max(0.0))
}

fn cstring(p: &Path) -> Result<CString> {
    let s = p
        .to_str()
        .ok_or_else(|| anyhow!("model path is not UTF-8: {}", p.display()))?;
    CString::new(s).map_err(Into::into)
}

/// The four files a streaming transducer needs.
///
/// Their names carry the training epoch and the chunk/left-context sizes, so
/// they differ between releases and cannot be hard-coded. Both a float and an
/// int8 export ship in the same directory; int8 is what task 1.0a measured
/// (179 MiB, WER 21.2-21.5% on the AMI clips) and what §8 locks in.
#[derive(Debug, PartialEq, Eq)]
struct ModelFiles {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
}

impl ModelFiles {
    fn find(dir: &Path) -> Result<Self> {
        let names: Vec<String> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        Ok(Self {
            encoder: dir.join(pick(&names, "encoder")?),
            decoder: dir.join(pick(&names, "decoder")?),
            joiner: dir.join(pick(&names, "joiner")?),
            tokens: {
                let t = dir.join("tokens.txt");
                if !t.is_file() {
                    bail!("no tokens.txt in {}", dir.display());
                }
                t
            },
        })
    }
}

fn pick(names: &[String], part: &str) -> Result<String> {
    let mut hits: Vec<&String> = names
        .iter()
        .filter(|n| n.starts_with(part) && n.ends_with(".int8.onnx"))
        .collect();
    if hits.is_empty() {
        hits = names
            .iter()
            .filter(|n| n.starts_with(part) && n.ends_with(".onnx"))
            .collect();
    }
    hits.sort();
    hits.first()
        .map(|s| (*s).clone())
        .ok_or_else(|| anyhow!("no {part}*.onnx in the model directory"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn int8_wins_when_both_exports_are_present() {
        let n = names(&[
            "encoder-epoch-99-avg-1.onnx",
            "encoder-epoch-99-avg-1.int8.onnx",
        ]);
        assert_eq!(
            pick(&n, "encoder").unwrap(),
            "encoder-epoch-99-avg-1.int8.onnx"
        );
    }

    #[test]
    fn a_float_only_release_still_loads() {
        let n = names(&["joiner-epoch-99-avg-1.onnx"]);
        assert_eq!(pick(&n, "joiner").unwrap(), "joiner-epoch-99-avg-1.onnx");
    }

    #[test]
    fn a_missing_part_names_the_part() {
        let n = names(&["encoder-epoch-99-avg-1.onnx", "tokens.txt"]);
        let err = pick(&n, "joiner").unwrap_err().to_string();
        assert!(err.contains("joiner"), "{err}");
    }

    #[test]
    fn the_result_json_parses_without_the_fields_we_ignore() {
        let raw = r#"{"text":"HELLO","tokens":["▁HELLO"],"timestamps":[0.64],
                      "ys_probs":[-0.1],"lm_probs":[],"context_scores":[],
                      "segment":2,"start_time":12.8,"is_final":false}"#;
        let r: SherpaResult = serde_json::from_str(raw).unwrap();
        assert_eq!(r.tokens, ["▁HELLO"]);
        assert_eq!(r.start_time, 12.8);
    }
}
