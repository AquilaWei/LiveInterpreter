//! Types shared by every stage of the pipeline (PLAN §10.1).
//!
//! Nothing here depends on a backend, a runtime, or a platform, so the crates
//! either side of an interface can agree on a vocabulary without depending on
//! each other. `li-core` is the only crate that assembles them.
//!
//! Two conventions worth stating once, because both cost real time to discover:
//!
//! * **Two clocks.** [`Duration`] fields are positions on the *audio* timeline,
//!   measured from the start of the session. [`Instant`] fields are wall time,
//!   for latency measurement. Latency is "audio for this sentence *ends*" ->
//!   "text on screen" (PLAN §7), so mixing the two silently reports nonsense.
//! * **Sentences carry a real span.** `t_start` and `t_end` are the first and
//!   last word of the sentence. The Phase 0 PoC filled both with the same
//!   value and every SRT block came out zero-length.
//!
//! The one exception to "no platform" is [`paths`], which is *about* the
//! platform difference: it is here because three crates each need the same
//! answer and one of them had quietly got it wrong on Windows.

pub mod paths;

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// 16 kHz mono f32 PCM, already resampled and level-normalised by `li-audio`.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub t_capture: Instant,
}

/// A run of speech the VAD gate let through.
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub id: u64,
    pub samples: Vec<f32>,
    pub t_start: Duration,
    pub t_end: Duration,
}

/// One recognised word with its position on the audio timeline.
///
/// `li-stream` needs per-word times to align the two lanes, to apply the word-age
/// commit rule, and to give `li-transcript` a real sentence span.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: Duration,
    pub end: Duration,
}

/// Which engine produced a piece of text.
///
/// This reaches the transcript file, not just the screen: a line that came from
/// the fast lane has a higher error rate, and a reader of the `.jsonl` needs to
/// be able to tell.
///
/// Since task 1.25 it also has punctuation and casing -- which is why the flag
/// matters more, not less. The marks are **restored by a model that reads the
/// words**, not heard; on the user's own sessions that model puts a boundary in
/// the right place 84% of the times it opens its mouth, and finds 58% of the
/// ones that are there. Good enough to read, not evidence of what was said. The
/// accurate lane's marks come from whisper, which heard the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lane {
    Fast,
    Accurate,
}

/// Why a line ended up attributed to the fast lane (PLAN §12.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastReason {
    /// The accurate lane did not deliver within `promote_after_s`.
    AccurateTimeout,
    /// The accurate lane delivered, but materially shorter than the fast lane --
    /// whisper drops whole clauses when unsure, and the hole must not reach the
    /// transcript file (gate G2b; measured in task 1.0c).
    AccurateTruncated,
    /// The accurate lane delivered, but materially *longer* than the fast lane
    /// and repeating itself. Whisper's decoder gets stuck in a loop and emits
    /// the same clause several times over (task 1.4 reproduced it on demand by
    /// cropping the encoder context: content WER 15.2% -> 36.1%). The fast lane
    /// never loops, so its word count is the reference either way.
    AccurateLooped,
    /// The accurate lane is switched off (no GPU, or phone power-saving mode).
    AccurateDisabled,
}

/// Output of one ASR backend. `Partial` is overwritten; `Final` never changes.
#[derive(Debug, Clone)]
pub enum AsrEvent {
    Partial {
        seg_id: u64,
        text: String,
        /// The same hypothesis with per-word times, when the backend can give
        /// them cheaply. `li-stream` aligns the two lanes on the audio
        /// timeline, so the accurate lane always fills this; the fast lane
        /// leaves it empty, because its partials only ever reach the screen
        /// (PLAN §8.2) and timing every one of them would cost a decode round
        /// trip 30 times a second for a line that is about to be replaced.
        words: Vec<Word>,
    },
    Final {
        seg_id: u64,
        words: Vec<Word>,
        t_start: Duration,
        t_end: Duration,
    },
}

/// What the UI and the FFI layer observe.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Fast-lane text, shown dimmed. Replaced in place, never appended.
    SourcePartial {
        line_id: u64,
        text: String,
        lane: Lane,
        /// `Some(t_end)` once the fast lane has stopped revising this text: it
        /// reached its own endpoint, closed the line, and fixed its span. The
        /// value is where the line's audio ends, which is what latency is
        /// measured from (PLAN §7). `None` while the words are still moving.
        ///
        /// Three things read it. A front end can stop treating the line as a
        /// half-finished hypothesis -- show it from the start rather than
        /// keeping the tail. `li-core` hands it to the translator as a draft,
        /// worth about half a second on screen (PLAN §12.4); a partial that is
        /// still being revised must never go there, because they arrive dozens
        /// of times a second. And the same `t_end` dates the draft, so its
        /// latency is comparable with the settled line's.
        closed: Option<Duration>,
    },
    /// Accurate-lane text (or a promoted fast-lane line). Overwrites the same
    /// `line_id`, and is what reaches the transcript file and the translator.
    SourceFinal {
        line_id: u64,
        text: String,
        t_start: Duration,
        t_end: Duration,
        lane: Lane,
        reason: Option<FastReason>,
    },
    /// A translation for `line_id`, replacing whatever that line had before.
    ///
    /// `settled` is false for the draft made from the fast lane's text, which
    /// is on screen while the accurate lane is still working and is replaced
    /// by the settled one a second or so later. Only the settled translation
    /// reaches the transcript file: a draft is screen-only, for the same
    /// reason a fast-lane partial is.
    Translation {
        line_id: u64,
        text: String,
        settled: bool,
    },
    Status(EngineStatus),
    Metrics(LatencySample),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum EngineStatus {
    ModelLoading {
        what: String,
    },
    Running,
    Paused,
    /// A cloud backend dropped; `li-core` falls back to local if one is loaded.
    Reconnecting {
        backend: String,
    },
    Error {
        message: String,
    },
    Stopped,
}

impl EngineStatus {
    /// One sentence for a status line. Here rather than in each front end so
    /// the terminal and the floating bar cannot drift apart about what the
    /// engine is doing.
    pub fn describe(&self) -> String {
        match self {
            EngineStatus::ModelLoading { what } => format!("loading {what}…"),
            EngineStatus::Running => "listening".into(),
            EngineStatus::Paused => "paused".into(),
            EngineStatus::Reconnecting { backend } => format!("reconnecting to {backend}…"),
            EngineStatus::Error { message } => format!("error: {message}"),
            EngineStatus::Stopped => "stopped".into(),
        }
    }
}

/// [`EngineEvent`] as it crosses into a user interface.
///
/// A separate type because [`EngineEvent`] carries [`Duration`] and is matched
/// on exhaustively by the pipeline, and neither survives contact with a
/// front end: JSON has no duration, and adding a variant must not be a breaking
/// change for a UI that ignores it. Seconds as `f64`, one flat `kind` tag,
/// and the same shape for Tauri's `emit` and for the UniFFI callbacks of
/// Phase 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UiEvent {
    /// Fast-lane text. Shown dimmed and replaced in place, never appended.
    Partial {
        line_id: u64,
        text: String,
        lane: Lane,
        /// See [`EngineEvent::SourcePartial::settled`]: the words have stopped
        /// changing even though the accurate lane has not answered yet.
        settled: bool,
    },
    Final {
        line_id: u64,
        text: String,
        start_s: f64,
        end_s: f64,
        lane: Lane,
        reason: Option<FastReason>,
    },
    Translation {
        line_id: u64,
        text: String,
        /// False while this is the draft translated from fast-lane text. It
        /// will be replaced under the same `line_id`.
        settled: bool,
    },
    Status {
        /// Both, deliberately: `state` is what a UI branches on, `text` is what
        /// it shows without having to reimplement [`EngineStatus::describe`].
        #[serde(flatten)]
        status: EngineStatus,
        text: String,
    },
    Metrics {
        line_id: u64,
        lane: Lane,
        stage: Stage,
        settled: bool,
        latency_s: f64,
    },
}

impl From<&EngineEvent> for UiEvent {
    fn from(ev: &EngineEvent) -> Self {
        match ev {
            EngineEvent::SourcePartial {
                line_id,
                text,
                lane,
                closed,
            } => UiEvent::Partial {
                line_id: *line_id,
                text: text.clone(),
                lane: *lane,
                // A front end needs to know *whether* the words are settled,
                // never where the audio ended -- that is a measurement, and
                // `Metrics` is where measurements cross this boundary.
                settled: closed.is_some(),
            },
            EngineEvent::SourceFinal {
                line_id,
                text,
                t_start,
                t_end,
                lane,
                reason,
            } => UiEvent::Final {
                line_id: *line_id,
                text: text.clone(),
                start_s: t_start.as_secs_f64(),
                end_s: t_end.as_secs_f64(),
                lane: *lane,
                reason: *reason,
            },
            EngineEvent::Translation {
                line_id,
                text,
                settled,
            } => UiEvent::Translation {
                line_id: *line_id,
                text: text.clone(),
                settled: *settled,
            },
            EngineEvent::Status(s) => UiEvent::Status {
                status: s.clone(),
                text: s.describe(),
            },
            EngineEvent::Metrics(m) => UiEvent::Metrics {
                line_id: m.line_id,
                lane: m.lane,
                stage: m.stage,
                settled: m.settled,
                latency_s: m.latency.as_secs_f64(),
            },
        }
    }
}

/// Which row a [`LatencySample`] is about.
///
/// The two are answers to different questions a reader asks -- "when do the
/// words appear" and "when do I get to read them in my own language" -- and
/// they have different causes, so a number that mixed them would be useless
/// for finding out which half is slow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// The line was settled: PLAN §16's G1 (fast lane) and G2 (accurate).
    Source,
    /// A translation reached the screen for it -- the draft or the settled one,
    /// told apart by `settled`.
    Translation,
}

/// One latency observation, kept per lane because the two are held to different
/// gates: the fast lane owns G1, the accurate lane owns G2 (PLAN §16).
#[derive(Debug, Clone, Copy)]
pub struct LatencySample {
    pub line_id: u64,
    pub lane: Lane,
    pub stage: Stage,
    /// False for a translation drafted from fast-lane text. Always true for
    /// [`Stage::Source`], which has no draft.
    pub settled: bool,
    /// End of this sentence's audio -> text on screen.
    ///
    /// PLAN §7 pins the start of that interval: the moment the audio for the
    /// sentence *ends*, not the moment it began. Both are wall clock, not
    /// audio positions -- see the two-clocks note at the top of this file.
    pub latency: Duration,
}

/// One line as written to the transcript (PLAN §2.6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptLine {
    pub line_id: u64,
    pub start_s: f64,
    pub end_s: f64,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translation: Option<String>,
    pub lane: Lane,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<FastReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    /// A monitor/loopback node rather than a real input.
    pub is_loopback: bool,
}

/// Which input to capture from.
///
/// Serialised as a plain string -- `"system"`, `"mic"`, or a device id -- and
/// not as a tagged table, because this is what `[audio] source` in
/// `config.toml` holds and a settings window has to be able to rewrite it
/// without leaving the previous variant's `id` key behind. It is also the same
/// spelling the CLI's `--source` takes, so there is one thing to remember.
///
/// PLAN §14 had `source` plus a separate `device_id`; one string says the same
/// and cannot express the contradiction of `source = "mic"` with a device id
/// set. A device genuinely named `system` or `mic` would be shadowed, which no
/// PulseAudio node name (`alsa_output...monitor`) or WASAPI id looks like.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSelector {
    /// Whatever the OS is playing (PipeWire monitor / WASAPI loopback).
    SystemLoopback,
    Microphone,
    Device(String),
}

impl std::fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SystemLoopback => f.write_str("system"),
            Self::Microphone => f.write_str("mic"),
            Self::Device(id) => f.write_str(id),
        }
    }
}

impl DeviceSelector {
    /// Never fails: an unrecognised word is a device id, which is the only
    /// thing it could be. Whether that device exists is the capture backend's
    /// question, and it answers it with the list of the ones that do.
    pub fn from_name(s: &str) -> Self {
        match s {
            "system" | "loopback" => Self::SystemLoopback,
            "mic" | "microphone" => Self::Microphone,
            id => Self::Device(id.to_owned()),
        }
    }
}

impl Serialize for DeviceSelector {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for DeviceSelector {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self::from_name(&String::deserialize(d)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_selector_is_one_string_both_ways() {
        // It is the value of `[audio] source` and of `--source`, and a
        // settings window rewrites it in place -- so what it serialises to has
        // to be exactly what a person would type.
        for (text, sel) in [
            ("system", DeviceSelector::SystemLoopback),
            ("mic", DeviceSelector::Microphone),
            (
                "alsa_output.pci-0000_00_1f.3.monitor",
                DeviceSelector::Device("alsa_output.pci-0000_00_1f.3.monitor".into()),
            ),
        ] {
            assert_eq!(serde_json::to_string(&sel).unwrap(), format!("\"{text}\""));
            assert_eq!(
                serde_json::from_str::<DeviceSelector>(&format!("\"{text}\"")).unwrap(),
                sel
            );
        }
        // The CLI's other spellings read back as the canonical ones.
        assert_eq!(
            DeviceSelector::from_name("loopback"),
            DeviceSelector::SystemLoopback
        );
        assert_eq!(
            DeviceSelector::from_name("microphone"),
            DeviceSelector::Microphone
        );
    }
}
