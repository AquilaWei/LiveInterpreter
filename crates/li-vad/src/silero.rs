//! Silero VAD v5 through ONNX Runtime.
//!
//! The model is embedded in the binary. At 2 MB it is the one model small
//! enough to ship (PLAN §15), and embedding it removes the whole class of
//! "works here, no model on the user's machine" failures for the component
//! that decides whether the recogniser runs at all.

use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use li_types::AudioFrame;
use ort::{session::Session, value::Tensor};

use crate::{
    Vad, VadEvent,
    gate::{Gate, GateConfig},
};

/// silero-vad v5 (MIT, snakers4/silero-vad), `src/silero_vad/data/silero_vad.onnx`.
const MODEL: &[u8] = include_bytes!("../../../assets/models/silero_vad.onnx");

/// Samples the model scores at once. Fixed by the model at 16 kHz.
pub const WINDOW: usize = 512;
/// Samples of history prepended to each window.
///
/// This is not optional and not internal to the graph. Feed the model a bare
/// 512-sample window and it does not fail -- it returns ~0.001 for every
/// window of clear speech, i.e. it silently reports that nobody is talking.
const CONTEXT: usize = 64;
/// The LSTM state, shaped `[2, 1, 128]`.
const STATE: usize = 2 * 128;

const SAMPLE_RATE: u32 = 16_000;
const WINDOW_DUR: Duration = Duration::from_nanos((WINDOW as u64 * 1_000_000_000) / 16_000);

pub struct SileroVad {
    session: Session,
    gate: Gate,
    /// Samples that have arrived but do not yet fill a window.
    pending: Vec<f32>,
    /// `CONTEXT` history samples followed by the window being scored, reused
    /// across calls so the hot path allocates nothing.
    input: Vec<f32>,
    state: Vec<f32>,
}

impl SileroVad {
    pub fn new(cfg: GateConfig) -> Result<Self> {
        Self::from_bytes(MODEL, cfg)
    }

    /// Load a model from disk instead of the embedded one -- for trying a new
    /// Silero release without a rebuild.
    pub fn from_path(path: impl AsRef<std::path::Path>, cfg: GateConfig) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).with_context(|| format!("reading VAD model {}", path.display()))?;
        Self::from_bytes(&bytes, cfg)
    }

    fn from_bytes(bytes: &[u8], cfg: GateConfig) -> Result<Self> {
        // One thread: the model is tiny, and the cores matter to the two
        // recognisers sharing this machine (PLAN §11).
        // `with_intra_threads` hands the builder back inside its error, which
        // makes the error type unusable with `?`; flatten it to a message.
        let mut builder = Session::builder()?
            .with_intra_threads(1)
            .map_err(|e| anyhow!("configuring the VAD session: {e}"))?;
        let session = builder
            .commit_from_memory(bytes)
            .context("loading the Silero VAD model")?;
        Ok(Self {
            session,
            gate: Gate::new(cfg),
            pending: Vec::with_capacity(WINDOW * 2),
            input: vec![0.0; CONTEXT + WINDOW],
            state: vec![0.0; STATE],
        })
    }

    pub fn probability(&self) -> f32 {
        self.gate.probability()
    }

    /// Forget the audio history. Call it when capture restarts on a new device;
    /// the LSTM state from the old stream is meaningless against the new one.
    pub fn reset(&mut self) {
        self.gate.reset();
        self.pending.clear();
        self.input.fill(0.0);
        self.state.fill(0.0);
    }

    /// Score one window, advancing the model state. `window.len() == WINDOW`.
    fn score(&mut self, window: &[f32]) -> Result<f32> {
        self.input[CONTEXT..].copy_from_slice(window);

        let outputs = self.session.run(ort::inputs![
            "input" => Tensor::from_array((vec![1_i64, self.input.len() as i64], self.input.clone()))?,
            "state" => Tensor::from_array((vec![2_i64, 1, 128], self.state.clone()))?,
            "sr" => Tensor::from_array((Vec::<i64>::new(), vec![i64::from(SAMPLE_RATE)]))?,
        ])?;

        let (_, prob) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob = *prob.first().context("VAD returned an empty output")?;
        let (_, next) = outputs["stateN"].try_extract_tensor::<f32>()?;
        self.state.copy_from_slice(next);

        // The tail of this window is the next window's context.
        self.input.copy_within(WINDOW.., 0);
        Ok(prob)
    }
}

impl Vad for SileroVad {
    fn push(&mut self, frame: &AudioFrame) -> Result<Vec<VadEvent>> {
        if frame.sample_rate != SAMPLE_RATE {
            bail!(
                "VAD needs {SAMPLE_RATE} Hz, got {} -- li-audio should have resampled",
                frame.sample_rate
            );
        }
        self.pending.extend_from_slice(&frame.pcm);

        let mut events = Vec::new();
        let mut consumed = 0;
        while self.pending.len() - consumed >= WINDOW {
            // Copy the window out so `score` can borrow self mutably. One
            // 2 KB copy per 32 ms of audio is not worth an unsafe split.
            let window: [f32; WINDOW] = self.pending[consumed..consumed + WINDOW]
                .try_into()
                .expect("slice is WINDOW long");
            consumed += WINDOW;
            let prob = self.score(&window)?;
            events.extend(self.gate.push(prob, WINDOW_DUR));
        }
        self.pending.drain(..consumed);
        Ok(events)
    }

    fn is_speech(&self) -> bool {
        self.gate.is_speech()
    }

    fn silence(&self) -> Duration {
        self.gate.silence()
    }
}
