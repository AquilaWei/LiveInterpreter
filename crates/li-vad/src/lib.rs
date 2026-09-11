//! Silero VAD, used as a *gate* and nothing more.
//!
//! Phase 0 tried using the VAD to cut utterance boundaries and it mis-cut badly:
//! calling Silero per window resets its internal state. So it answers two
//! questions only — is there speech right now (skip ASR on silence, which also
//! stops Whisper hallucinating on music), and how long has the trailing silence
//! run (a sentence break). Segmentation belongs to `li-stream`.

pub mod gate;
pub mod silero;

use anyhow::Result;
use li_types::AudioFrame;

pub use gate::{Gate, GateConfig};
pub use silero::SileroVad;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    SpeechStart,
    SpeechEnd,
}

pub trait Vad: Send {
    fn push(&mut self, frame: &AudioFrame) -> Result<Vec<VadEvent>>;

    fn is_speech(&self) -> bool;

    /// Length of the current run of trailing silence.
    fn silence(&self) -> std::time::Duration;
}
