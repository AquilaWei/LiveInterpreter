//! ASR backends for both lanes (PLAN §8.2).
//!
//! The two lanes exist because one engine cannot hold both acceptance gates.
//! The fast lane is chunk-based and never re-transcribes, so its latency does
//! not grow with the buffer — it owns G1 (measured: median 0.68–0.76 s). The
//! accurate lane transcribes a whole utterance once its endpoint has been seen,
//! lands about a second behind, and owns G2a — it is worth 5–7 WER points over
//! the fast lane (task 1.0c), which is what pays for running two engines.
//!
//! Note both lanes are `AsrEngine`: the difference is the commit policy in
//! `li-stream`, not the interface.
//!
//! ## Two things this crate does not decide
//!
//! * **When a hypothesis becomes text on screen.** [`AsrEngine::feed`] reports
//!   what the engine currently believes; `li-stream` decides what is stable.
//!   The fast lane is the exception only because its backend has its own
//!   endpoint detector, and throwing that away would mean re-deriving sentence
//!   breaks from unpunctuated text.
//! * **How much audio the accurate lane sees.** The caller owns the buffer.
//!   Letting the engine grow its own is what broke the Phase 0 PoC (task 1.0b),
//!   and task 1.5 measured that *where* the buffer starts matters more than how
//!   long it is: whisper handed a window beginning mid-clause condenses and
//!   repeats itself. `li_stream::StreamConfig::max_segment_s` is the knob, and
//!   one endpoint-to-endpoint utterance per pass is the policy (PLAN §12.3).

use std::{path::PathBuf, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use li_types::{AsrEvent, Word};
use serde::{Deserialize, Serialize};

pub mod device;
pub mod window;
pub mod words;

#[cfg(feature = "sherpa")]
pub mod punct;
#[cfg(feature = "sherpa")]
pub mod sherpa;
#[cfg(feature = "whispercpp")]
pub mod whisper;

pub use device::{Accel, BackendInfo, DeviceRequest, GpuInfo, Selection};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsrCaps {
    /// True for a chunk-based streaming model that reports its own endpoints.
    /// False means the caller must re-feed a growing buffer and rely on
    /// `li-stream` to decide what is stable.
    pub native_streaming: bool,
    /// Whether the output carries punctuation and casing. The fast lane's does
    /// not, so sentence breaking must not depend on it.
    pub punctuated: bool,
}

#[async_trait]
pub trait AsrEngine: Send {
    fn capabilities(&self) -> AsrCaps;

    /// What this engine actually loaded and what it ended up running on.
    /// PLAN §8.1 requires the real backend and device name to reach the status
    /// line, not the value that was asked for: `auto` resolving to CPU on a
    /// machine that was supposed to have a GPU is the failure worth seeing.
    fn backend(&self) -> &BackendInfo;

    /// Feed audio. For a streaming backend this is the next chunk; for a
    /// re-transcribing one it is the whole rolling buffer.
    ///
    /// `t_origin` is where `pcm[0]` sits on the audio timeline. It is not
    /// derivable from the samples fed so far: `li-audio` drops frames when the
    /// pipeline falls behind, and every timestamp downstream — the latency
    /// metric, the SRT block, the two-lane alignment — is on the wall timeline,
    /// not on the "audio that made it through" one.
    async fn feed(&mut self, pcm: &[f32], t_origin: Duration) -> Result<Vec<AsrEvent>>;

    /// Force the current segment to final, e.g. at end of capture. The engine
    /// owns segment numbering, so this takes no id.
    async fn finalize(&mut self) -> Result<Option<AsrEvent>>;

    /// Word times for the hypothesis that is currently open, without consuming
    /// it.
    ///
    /// Default `Vec::new()`, because most backends have no such thing to
    /// report: a re-transcribing lane re-reads its whole buffer every pass and
    /// has no notion of a sentence still in progress. Only a backend that
    /// segments its own stream can answer.
    ///
    /// Task 1.25 stage 2 asks this when the punctuation model says a sentence
    /// has ended inside a line that is still open. Closing that line needs a
    /// `t_end`, and the two values that are free are both wrong: `now` is the
    /// edge of the audio, which would make every early line report a latency of
    /// about zero and flatter the one gate the feature is judged by, and a
    /// value past the real boundary silently deletes the head of the next
    /// accurate-lane line (`li_stream::stream`'s `emitted_through`). So the
    /// answer has to be a real word onset, and this is where it comes from.
    fn open_words(&mut self) -> Vec<Word> {
        Vec::new()
    }

    /// Text the next pass should be biased toward: the tail of what has already
    /// been committed. Phase 0 measured that carrying it improves recognition
    /// at the start of a buffer, which for a re-transcribing backend is every
    /// pass (PLAN §12.3). A streaming backend never re-reads its input and
    /// ignores this.
    fn set_prompt(&mut self, _text: &str) {}
}

/// Trailing silence the fast lane takes as the end of a sentence.
///
/// 0.6 s, mirroring `li_stream::StreamConfig::pause_flush_s`. Measured on
/// `ami_meeting.wav`: it is 0.6 of the 0.674 s median that a finished fast-lane
/// sentence takes to reach the screen, so it is where the latency is.
///
/// Task 1.24 swept it again looking for the same latency the cap above bought,
/// and 0.45 -- untested by task 1.18, which jumped from 0.60 to 0.35 -- looks
/// free on read speech: `read_clean.wav` and `read_hard.wav` keep their word
/// error rate exactly. It is not free. On `ami_meeting.wav` it takes content
/// WER from 14.7% to **21.9%**, past G2a, and the extra 8 passes push a clip
/// that was already only just keeping up to 77% over its own length. Read
/// speech pauses cleanly between sentences; conversation hesitates at 0.45 s
/// in the middle of them. **Leave it at 0.6.**
pub const DEFAULT_ENDPOINT_SILENCE_S: f32 = 0.6;

/// Hard cap on one utterance: a speaker who never leaves
/// [`DEFAULT_ENDPOINT_SILENCE_S`] of silence still gets lines.
///
/// This is a give-up, not a sentence break — it lands wherever the clock says,
/// not where the speaker stopped — so the number is a bound on how bad the
/// give-up is allowed to be, and nothing else.
///
/// **12 s, measured (task 1.24).** It was 20 until a user reported fast speech
/// arriving too late to read, and 20 turned out to be where that happens: on
/// `read_clean.wav` resampled to 1.5x speed, one line ran the full 19.0-39.0 s
/// and the accurate lane timed out on it, so the bar showed unpunctuated
/// fast-lane text for a fifth of the clip.
///
/// | cap | G1 median / p90 | G3 | content WER |
/// |---|---|---|---|
/// | 20 | 0.71 / **2.71** | **5.27** | 11.7% |
/// | **12** | **0.67 / 1.56** | 4.87 | **11.7%** |
/// | 8 | 0.11 / 0.71 | 3.78 | 12.9% |
/// | 6 | 0.15 / **2.78** | 4.60 | **23.0%** |
///
/// 8 and 6 are better on the fast clip and worse everywhere else: at 8,
/// `read_clean.wav` at its own speed goes from 10 lines to 14 and G1's p90 from
/// 0.75 to 2.48 s; at 6 the accurate lane's load reaches 0.84 engine seconds
/// per audio second and whisper starts looping on the 4.7 s windows it is left
/// with. 12 is the largest cap that fixes the fast clip, and on
/// `read_clean.wav`, `read_hard.wav` and `ami_meeting.wav` at normal speed it
/// changes **nothing at all** -- same line count, same word error rate, because
/// no utterance in them reaches it.
pub const DEFAULT_MAX_UTTERANCE_S: f32 = 12.0;

/// One lane's configuration, resolved from `config.toml` by `li-core`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneSpec {
    /// `sherpa` | `whispercpp`.
    pub backend: String,
    /// A directory for sherpa (the three onnx files and `tokens.txt` inside it
    /// are named per release), a single `.bin` for whisper.cpp.
    pub model: PathBuf,
    pub device: DeviceRequest,
    pub threads: usize,
    /// BCP-47-ish, as whisper.cpp wants it. The fast lane's model is English-only.
    pub language: String,
    /// Trailing silence, in seconds, before the fast lane calls a sentence
    /// finished. Ignored by the accurate lane, whose utterance boundaries come
    /// from the fast lane's endpoints (PLAN §12.3).
    ///
    /// It is the single largest term in G1: a sentence cannot be closed sooner
    /// than this, so 0.6 is 0.6 of the ~0.67 s a finished line takes to reach
    /// the screen. Lowering it moves both the settled source line and the draft
    /// translation earlier by the difference, and cuts sentences at every
    /// hesitation -- which is why it is a setting and not a smaller constant.
    pub endpoint_silence_s: f32,
    /// Seconds after which the fast lane closes a line whatever the speaker is
    /// doing. See [`DEFAULT_MAX_UTTERANCE_S`]; the accurate lane ignores it, as
    /// it does `endpoint_silence_s`.
    pub max_utterance_s: f32,
}

impl LaneSpec {
    pub fn new(backend: &str, model: impl Into<PathBuf>) -> Self {
        Self {
            backend: backend.to_owned(),
            model: model.into(),
            device: DeviceRequest::Auto,
            threads: 2,
            language: "en".into(),
            endpoint_silence_s: DEFAULT_ENDPOINT_SILENCE_S,
            max_utterance_s: DEFAULT_MAX_UTTERANCE_S,
        }
    }
}

/// Build the engine a `LaneSpec` names.
///
/// A backend that exists but was compiled out is a different error from one
/// that does not exist, and the message says which feature would bring it back
/// — otherwise a stripped-down build looks like a typo in the config file.
pub fn build(spec: &LaneSpec) -> Result<Box<dyn AsrEngine>> {
    match spec.backend.as_str() {
        #[cfg(feature = "sherpa")]
        "sherpa" => Ok(Box::new(sherpa::SherpaFast::open(spec)?)),
        #[cfg(not(feature = "sherpa"))]
        "sherpa" => anyhow::bail!("this build has no fast lane: rebuild with --features sherpa"),

        #[cfg(feature = "whispercpp")]
        "whispercpp" => Ok(Box::new(whisper::WhisperAccurate::open(spec)?)),
        #[cfg(not(feature = "whispercpp"))]
        "whispercpp" => {
            anyhow::bail!("this build has no accurate lane: rebuild with --features whispercpp")
        }

        other => anyhow::bail!("unknown asr backend {other:?} (want `sherpa` or `whispercpp`)"),
    }
}
