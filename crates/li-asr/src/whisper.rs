//! Accurate lane: whisper.cpp / GGML `small.en` q5_1.
//!
//! Measured: offline RTF 0.053 on Vulkan against 0.123 on the CPU,
//! WER within 1.2 points of `medium` for a third of the compute. `base.en` is
//! the no-GPU fallback and is a config change, not a code path.
//!
//! ## This lane does not decide anything
//!
//! Every [`AsrEngine::feed`] transcribes the buffer it is handed and returns
//! the whole hypothesis, words and all. It is `li-stream` that decides which
//! audio a pass gets and what happens to the result. That division is the
//! lesson of the Python prototype -- its collapse was a buffer that grew to ~22 s
//! and fed back on itself, not an engine that was too slow -- and measurement
//! sharpened it: one pass over one whole utterance, cut at the fast lane's own
//! endpoints, beat every re-transcription policy tried.
//!
//! ## Two parameter choices that are latency decisions
//!
//! * `no_context` — each pass starts clean. Carrying decoder context between
//!   passes over overlapping audio is how whisper starts repeating itself, and
//!   a repetition loop in a subtitle bar is worse than a missed word.
//! * **The temperature fallback stays on** (`temperature_inc` at whisper's own
//!   default). It re-decodes a segment at a higher temperature when the result
//!   looks degenerate. Turning it off looks like free latency and is not quite:
//!   measured on `ami_meeting2` it costs about 4% of pass time and buys about
//!   half a point of content WER (500 → 520 ms, 15.7% → 15.2%). Cheap, on a
//!   lane with this much headroom. It is *not* a repetition guard — see below.
//! * **The encoder context is never cropped.** See [`audio_ctx_for`] for what
//!   was tried, what it cost, and why it is not the default anywhere.

use std::{path::Path, time::Duration};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use li_types::AsrEvent;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::{
    AsrCaps, AsrEngine, LaneSpec,
    device::{self, Accel, BackendInfo},
    words,
};

/// whisper.cpp reports segment and token times in centiseconds.
const TIME_UNIT: Duration = Duration::from_millis(10);

/// Beam 5 is what was measured. Lower would be faster and is a knob to turn
/// if the soak test needs it; changing it silently would invalidate the WER
/// numbers the acceptance gates sign off on.
const BEAM_SIZE: i32 = 5;

/// whisper's encoder context, in units of 20 ms, for a full 30 s window.
const FULL_AUDIO_CTX: i32 = 1500;
/// Cropping the encoder below this starts to cost real accuracy: the model's
/// positional embeddings were trained at full length, and a very short context
/// makes it drop the end of the window. Roughly 10 s.
const MIN_AUDIO_CTX: i32 = 512;
/// Encoder frames per second of audio (1500 / 30).
const CTX_PER_SECOND: f64 = 50.0;

/// whisper.cpp's temperature-fallback step. This is whisper's own default and
/// it is deliberately not zero: see the note at the top of this file.
const TEMPERATURE_INC: f32 = 0.2;

/// How much encoder context this buffer needs.
///
/// **whisper always encodes a 30-second window.** It pads whatever it is given
/// up to 30 s and runs the encoder over all of it, so a 4-second buffer costs
/// exactly as much as a 30-second one. That is why the offline RTF from task
/// The first benchmark — measured over whole files, where the padding is amortised — does not
/// predict the streaming cost: at a 0.8 s tick, a 12 s rolling buffer means
/// ~1.2 encodes of a 30 s window *per second of audio*, and the lane cannot
/// keep up on any hardware.
///
/// `audio_ctx` shortens the encoder to the audio actually present, which is
/// what whisper.cpp's own streaming example does. One second of slack is added
/// so the tail of the buffer is never at the very edge of the context.
///
/// # This is measured, and rejected. Do not turn it on.
///
/// It halves the cost of a pass, and it makes whisper **repeat itself**. On
/// `ami_meeting2` a cropped run produced
///
/// > …and then when you go on the menu, you can select the description box,
/// > and then when you go on the menu, you can select the description box,
/// > and then when you go on the menu, you can select the description…
///
/// Measured on Arc 140V, one run per cell:
///
/// | encoder context | ms/pass | content WER `ami_meeting` | `ami_meeting2` |
/// |---|---|---|---|
/// | full 30 s | 615 | **13.8%** | **15.2%** |
/// | cropped | 301 | 17.0% | **36.1%** (loops) |
///
/// The temperature fallback does not save it: cropped-with-fallback still
/// loops (36.1%), and cropped-without is the same (35.7%). Starving the
/// encoder is what does it, so no decoder-side guard can undo it.
///
/// The tempting use was the no-GPU path, where the full window does not fit in
/// the clock. **The answer there is a smaller model, not a shorter encoder** —
/// which is the fallback the design already chose: `base.en` rather than a crippled
/// `small.en`. This function stays public because the trap is worth being able
/// to reproduce, not because anything should call it.
pub fn audio_ctx_for(buffer: Duration) -> i32 {
    let needed = ((buffer.as_secs_f64() + 1.0) * CTX_PER_SECOND).ceil() as i32;
    needed.clamp(MIN_AUDIO_CTX, FULL_AUDIO_CTX)
}

pub struct WhisperAccurate {
    /// Also read for `token_eot`. Held regardless: the state borrows it through
    /// an `Arc`, so dropping the context would take the model out from under it.
    ctx: WhisperContext,
    state: WhisperState,
    info: BackendInfo,
    language: String,
    threads: i32,
    seg_id: u64,
    /// Words from the most recent pass, kept so `finalize` can hand back the
    /// last hypothesis without re-running the model.
    last: Vec<li_types::Word>,
    /// Crop the encoder to the buffer length. Off, on every device; the switch
    /// exists so [`audio_ctx_for`]'s failure stays reproducible.
    crop_ctx: bool,
    /// whisper's temperature fallback, i.e. its repetition guard. On by
    /// default; the switch exists so the cost of turning it off stays
    /// measurable rather than becoming folklore.
    fallback: bool,
    /// The tail of the committed transcript, set by `li-stream` between passes.
    prompt: String,
}

impl WhisperAccurate {
    pub fn open(spec: &LaneSpec) -> Result<Self> {
        let path: &Path = &spec.model;
        if !path.is_file() {
            bail!(
                "accurate-lane model not found: {}\n\
                 Fetch a GGML .bin or point `[asr.accurate] model` at one.",
                path.display()
            );
        }
        // whisper.cpp and ggml print their banner and every warning straight to
        // stderr. A subtitle app has no terminal to print them to, and the
        // Vulkan device line is exactly what we do want to keep -- so redirect
        // the lot into `tracing` instead of losing it.
        static HOOKS: std::sync::Once = std::sync::Once::new();
        HOOKS.call_once(whisper_rs::install_logging_hooks);

        let selection = device::resolve(spec.device);
        if let Some(note) = &selection.note {
            tracing::warn!("accurate lane: {note}");
        }

        let mut params = WhisperContextParameters::default();
        params.use_gpu(selection.accel != Accel::Cpu);
        params.gpu_device(selection.gpu_index);
        let (ctx, state) = {
            // One load at a time, process-wide. Two models loading onto Vulkan
            // at once segfault inside ggml (`ggml_backend_alloc_ctx_tensors_from_buft`
            // calls a null function pointer): the backend's initialisation is
            // not thread-safe. Measured 3 crashes in 3 runs with four loads in
            // parallel; see `four_accurate_lanes_can_load_at_the_same_time`.
            // Loads take seconds and almost never overlap, so the wait is free.
            static LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());
            // A load that panicked leaves nothing behind that the next one
            // depends on, so a poisoned lock is still a usable one.
            let _one = LOAD.lock().unwrap_or_else(|e| e.into_inner());
            let ctx = WhisperContext::new_with_params(path, params)
                .with_context(|| format!("loading {}", path.display()))?;
            let state = ctx.create_state().context("creating a whisper state")?;
            (ctx, state)
        };

        let info = BackendInfo {
            engine: "whisper.cpp",
            model: path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
            selection,
        };
        tracing::info!("accurate lane: {info}");

        Ok(Self {
            ctx,
            state,
            info,
            language: spec.language.clone(),
            threads: spec.threads.max(1) as i32,
            seg_id: 0,
            last: Vec::new(),
            crop_ctx: false,
            fallback: true,
            prompt: String::new(),
        })
    }

    /// Override the encoder-context policy. For measurement.
    pub fn with_cropped_context(mut self, on: bool) -> Self {
        self.crop_ctx = on;
        self
    }

    /// Override the temperature fallback. For measurement.
    pub fn with_temperature_fallback(mut self, on: bool) -> Self {
        self.fallback = on;
        self
    }

    /// Free function, not a method: `FullParams` borrows the language string,
    /// which would keep `&self` alive across the `&mut self.state` call below.
    fn params<'a>(
        language: &'a str,
        threads: i32,
        audio_ctx: i32,
        fallback: bool,
        prompt: &str,
    ) -> FullParams<'a, 'a> {
        let mut p = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: BEAM_SIZE,
            patience: 0.0,
        });
        p.set_n_threads(threads);
        p.set_audio_ctx(audio_ctx);
        p.set_language(Some(language));
        p.set_translate(false);
        p.set_token_timestamps(true);
        p.set_no_context(true);
        p.set_temperature(0.0);
        p.set_temperature_inc(if fallback { TEMPERATURE_INC } else { 0.0 });
        p.set_print_special(false);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_timestamps(false);
        if !prompt.is_empty() {
            // Upstream wart: `set_initial_prompt` leaks the CString it builds
            // (`into_raw`, never reclaimed), so this costs a couple of hundred
            // bytes per pass -- under 2 MB an hour at the 0.8 s tick, against a
            // 3.5 GB budget. Worth knowing about before reading a G4 soak
            // graph; not worth a fork. Null bytes would panic, so they go.
            p.set_initial_prompt(&prompt.replace('\0', " "));
        }
        p
    }

    /// Read the pass just run into words on the session timeline.
    ///
    /// `limit` is where the audio it was given ends. whisper pads its input to
    /// a 30 s window and will happily report a segment that runs past the real
    /// end of the buffer, so times are clamped there. Letting one through is
    /// not cosmetic: `li-stream` trims the rolling buffer to the commit
    /// frontier, so a word timed in the future moves the buffer origin into the
    /// future, every later word inherits the error, the buffer-length check
    /// underflows to zero and the cap never fires again. It was measured that
    /// as a lane running 12 s behind with a *negative* reported latency.
    fn collect(&self, t_origin: Duration, limit: Duration) -> Result<Vec<li_types::Word>> {
        let mut tokens: Vec<(Vec<u8>, Duration)> = Vec::new();
        let mut t_end = t_origin;
        for seg in self.state.as_iter() {
            if seg.to_str_lossy().is_ok_and(|t| is_annotation(&t)) {
                // whisper writes non-speech as ordinary text, not as a special
                // token: "[BLANK_AUDIO]", "(upbeat music)", "[ Silence ]".
                // Nothing downstream would strip it, so it would reach the
                // subtitle bar and the transcript file that has to be a
                // record of what was said.
                continue;
            }
            let seg_end = (t_origin + TIME_UNIT * seg.end_timestamp().max(0) as u32).min(limit);
            t_end = t_end.max(seg_end);
            for i in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(i) else {
                    continue;
                };
                // Timestamp and control tokens sit above `token_eot` and carry
                // no text; printing them would put "[_BEG_]" in the subtitle.
                if tok.token_id() >= self.ctx.token_eot() {
                    continue;
                }
                // Bytes, not text: a token can end halfway through a
                // character, and decoding it alone ruins both halves.
                let Ok(bytes) = tok.to_bytes() else {
                    continue;
                };
                let start = (t_origin + TIME_UNIT * tok.token_data().t0.max(0) as u32).min(limit);
                tokens.push((bytes.to_vec(), start));
            }
        }
        Ok(words::merge(&words::decode_pieces(&tokens), t_end))
    }
}

#[async_trait]
impl AsrEngine for WhisperAccurate {
    fn capabilities(&self) -> AsrCaps {
        AsrCaps {
            native_streaming: false,
            punctuated: true,
        }
    }

    fn backend(&self) -> &BackendInfo {
        &self.info
    }

    async fn feed(&mut self, pcm: &[f32], t_origin: Duration) -> Result<Vec<AsrEvent>> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        let (language, threads) = (self.language.clone(), self.threads);
        let prompt = self.prompt.clone();
        let audio_ctx = if self.crop_ctx {
            audio_ctx_for(Duration::from_secs_f64(pcm.len() as f64 / 16_000.0))
        } else {
            FULL_AUDIO_CTX
        };
        self.state
            .full(
                Self::params(&language, threads, audio_ctx, self.fallback, &prompt),
                pcm,
            )
            .context("whisper.cpp transcription failed")?;
        let limit = t_origin + Duration::from_secs_f64(pcm.len() as f64 / 16_000.0);
        self.last = self.collect(t_origin, limit)?;
        if self.last.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![AsrEvent::Partial {
            seg_id: self.seg_id,
            text: words::text_of(&self.last),
            words: self.last.clone(),
        }])
    }

    fn set_prompt(&mut self, text: &str) {
        self.prompt = text.to_owned();
    }

    async fn finalize(&mut self) -> Result<Option<AsrEvent>> {
        let words = std::mem::take(&mut self.last);
        let (Some(first), Some(last)) = (words.first(), words.last()) else {
            return Ok(None);
        };
        let ev = AsrEvent::Final {
            seg_id: self.seg_id,
            t_start: first.start,
            t_end: last.end,
            words,
        };
        self.seg_id += 1;
        Ok(Some(ev))
    }
}

/// Is this whole segment one of whisper's non-speech annotations?
///
/// Only a segment that is *entirely* bracketed counts. A bracket inside real
/// speech is real text, and the transcript file should keep it.
fn is_annotation(text: &str) -> bool {
    let t = text.trim();
    matches!(
        (t.chars().next(), t.chars().last()),
        (Some('['), Some(']')) | (Some('('), Some(')'))
    ) && t.chars().count() > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_non_speech_annotations_are_not_speech() {
        for t in [
            "[BLANK_AUDIO]",
            " [ Silence ] ",
            "(upbeat music)",
            "[_TT_170]",
        ] {
            assert!(is_annotation(t), "{t:?}");
        }
    }

    #[test]
    fn a_bracket_inside_a_sentence_is_kept() {
        for t in [
            "the meeting (the first one) went well",
            "[laughs] and then we shipped it",
            "okay",
            "",
        ] {
            assert!(!is_annotation(t), "{t:?}");
        }
    }

    #[test]
    fn a_short_buffer_does_not_pay_for_thirty_seconds() {
        assert!(audio_ctx_for(Duration::from_secs(4)) < FULL_AUDIO_CTX);
    }

    #[test]
    fn context_grows_with_the_buffer() {
        assert!(audio_ctx_for(Duration::from_secs(12)) > audio_ctx_for(Duration::from_secs(6)));
    }

    #[test]
    fn the_context_covers_the_audio_plus_slack() {
        // 6 s of audio needs 300 frames; anything less and the tail of the
        // buffer -- the newest speech, the part the subtitle is waiting on --
        // falls outside the encoder.
        assert!(audio_ctx_for(Duration::from_secs(6)) >= 300);
    }

    #[test]
    fn a_long_buffer_is_capped_at_the_full_window() {
        assert_eq!(audio_ctx_for(Duration::from_secs(60)), FULL_AUDIO_CTX);
    }

    #[test]
    fn a_tiny_buffer_still_gets_a_usable_context() {
        assert_eq!(audio_ctx_for(Duration::from_millis(200)), MIN_AUDIO_CTX);
    }
}
