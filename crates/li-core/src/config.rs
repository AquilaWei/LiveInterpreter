//! `config.toml`.
//!
//! Read from `~/.config/liveinterpreter/config.toml`. A missing file is not an
//! error -- it means the defaults, which are the settings that were measured
//! and locked. A malformed one *is* an error, named and
//! refused at startup rather than silently half-applied.

use std::path::PathBuf;

use anyhow::{Context, Result};
use li_asr::{DeviceRequest, LaneSpec};
use li_transcript::TranscriptConfig;
use li_types::DeviceSelector;
use serde::{Deserialize, Serialize};

use crate::models::{Kind, Models};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    pub audio: AudioCfg,
    pub asr: AsrCfg,
    pub mt: MtCfg,
    pub transcript: TranscriptConfig,
    pub ui: UiCfg,
    pub hotkeys: HotkeyCfg,
}

/// Global hotkeys.
///
/// Global because the two that matter are unreachable otherwise. A bar with
/// click-through on receives no clicks, so nothing on it can turn click-through
/// off again; and pausing is wanted at the moment something private is about to
/// be said, which is not a moment to go hunting for a window.
///
/// An empty string turns one off. A combination another application already
/// holds cannot be registered -- the desktop shell says so at startup rather
/// than leaving a key that does nothing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HotkeyCfg {
    pub click_through: String,
    pub pause: String,
    pub settings: String,
}

impl Default for HotkeyCfg {
    fn default() -> Self {
        Self {
            click_through: "Ctrl+Alt+C".into(),
            pause: "Ctrl+Alt+P".into(),
            settings: "Ctrl+Alt+S".into(),
        }
    }
}

/// The floating bar's appearance and placement.
///
/// Here rather than in `li-desktop` because it belongs in the one config file
/// the settings window writes, and because an Android bar
/// needs the same numbers.
///
/// **`max_lines` is not here.** The first schema had one integer for both lines, and it
/// cannot say the thing that matters: the two lines want different limits. The
/// source is context and may be clipped -- the transcript file has all of it --
/// while the translation is the product and must not be. So there is a row
/// budget per line instead, and `show_source = false` for the reader who wants
/// the translation alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiCfg {
    pub font_size: f64,
    pub opacity: f64,
    pub position: BarPosition,
    /// Bar width as a percentage of the monitor's width (default 80%).
    pub width_pct: f64,
    /// Gap between the bar and the screen edge it is anchored to, in logical
    /// pixels. Zero puts it under a panel or a taskbar on most desktops.
    pub margin_px: f64,
    /// Rows each line may wrap to before it is clipped. A line runs to
    /// `StreamConfig::max_words` in the worst case, which no single row holds.
    pub source_rows: usize,
    pub target_rows: usize,
    /// How many translations stay on the bar: the newest at the bottom, the
    /// ones before it dimmed above it, rolling up like broadcast captions.
    /// Each may wrap to `target_rows`. 1 is the old single-row bar.
    ///
    /// 2 by default because one was not enough for a fast speaker: in two
    /// sessions (2026-09-24) 13 of 67 translations were replaced before they
    /// could be read at 7 characters a second, and with two kept, 4. A
    /// minimum reading time was tried on the same sessions instead and lost:
    /// the bar fell up to 5.3 s behind and still had to skip lines.
    pub target_lines: usize,
    /// Shortest time a line stays on the bar before a newer one may take it.
    /// One fast-lane endpoint can close several lines at once,
    /// and without this they flash past unread.
    pub min_dwell_ms: u64,
    pub show_source: bool,
    /// Whether clicks pass through the bar to whatever is behind it, from the
    /// moment it opens. Toggled by [`HotkeyCfg::click_through`]; saved so a bar
    /// that has found its place stays out of the way on the next run.
    pub click_through: bool,
    /// Where the bar was dragged to, in physical pixels from where
    /// `position` / `width_pct` / `margin_px` would have put it.
    ///
    /// A dragged bar has to survive its next resize: the window is re-placed
    /// every time the translation wraps to another row, and without this it
    /// would jump back to the middle a few times a minute. Kept as an offset
    /// rather than as a position so that a screen that changes resolution, or
    /// a bar moved from a 4K monitor to a laptop panel, still lands somewhere
    /// on the screen.
    pub offset_x: f64,
    pub offset_y: f64,
}

/// Which screen edge the bar is anchored to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BarPosition {
    #[default]
    Bottom,
    Top,
}

impl Default for UiCfg {
    fn default() -> Self {
        Self {
            font_size: 22.0,
            opacity: 0.85,
            position: BarPosition::Bottom,
            width_pct: 80.0,
            margin_px: 40.0,
            source_rows: 1,
            target_rows: 2,
            target_lines: 2,
            min_dwell_ms: 700,
            show_source: true,
            click_through: false,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioCfg {
    pub source: DeviceSelector,
}

/// Both lanes are optional, but at least one must be present.
/// Turning the accurate lane off is the no-GPU and phone power-saving mode;
/// turning the fast lane off falls back to single-lane behaviour at ~2 s latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AsrCfg {
    pub language: String,
    pub fast: Option<LaneCfg>,
    pub accurate: Option<LaneCfg>,
    /// Longest subtitle line, in words, before one utterance is broken into
    /// several lines. A width limit, so a line fits the bar's row budget.
    ///
    /// **It is not a latency knob**, however much it looks like one. All the
    /// pieces of one utterance are emitted at its endpoint, in the same
    /// instant, so cutting a long line makes more lines rather than earlier
    /// ones. Measured on `read_clean.wav`, 40 -> 12 words took the
    /// transcript from 10 lines to 24 and the translation row's mean gap from
    /// 3.81 s to 2.19 s -- and left **the longest gap at 10.4 s and the number
    /// of gaps over 5 s at 8, both unchanged**, because what the reader waits
    /// for is the next endpoint either way. It is also actively worse on the
    /// bar: three lines arriving together are three lines competing for one
    /// row, and `min_dwell_ms` lets the older two past unread.
    ///
    /// It does not change what either ASR lane sees -- the accurate lane is
    /// still handed one whole endpoint-to-endpoint utterance -- so it costs
    /// nothing in word error rate.
    #[serde(default = "max_line_words")]
    pub max_line_words: usize,
}

fn max_line_words() -> usize {
    li_stream::StreamConfig::default().max_words
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneCfg {
    pub backend: String,
    pub model: String,
    /// `auto` | `cpu` | `vulkan` | `cuda` | `sycl` | `hip`.
    ///
    /// Typed rather than a string so a misspelling is a startup error naming
    /// the valid values, not a setting that is silently ignored. Note `sycl`
    /// where the first plan said `openvino`: `whisper-rs` exposes no OpenVINO backend,
    /// and OpenVINO only ever accelerated whisper's encoder anyway.
    #[serde(default)]
    pub device: DeviceRequest,
    /// Inference threads. `0` means the value each lane was measured with:
    /// 2 for the fast lane, and everything but two cores for the accurate one.
    #[serde(default)]
    pub threads: usize,
    /// Seconds to wait for this lane before promoting the fast lane's text.
    /// Read from `[asr.accurate]`; meaningless on `[asr.fast]`,
    /// which is the lane being promoted.
    #[serde(default = "promote_after_s")]
    pub promote_after_s: f64,
    /// Trailing silence, in seconds, before the fast lane calls a sentence
    /// finished. Read from `[asr.fast]`; meaningless on `[asr.accurate]`,
    /// which takes its utterance boundaries from the fast lane.
    ///
    /// **This is where the source row's latency is** -- and lowering it is not
    /// free. Measured on `ami_meeting.wav` (100 s, 240 reference words), task
    /// 1.18:
    ///
    /// | silence | fast sentence on screen | 中文 on screen | lines | content WER |
    /// |---|---|---|---|---|
    /// | **0.60** | **673 ms** | **1052 ms** | **33** | **15.1%** |
    /// | 0.35 | 594 ms | 840 ms | 46 | 23.4% |
    /// | 0.25 | 354 ms | 675 ms | 58 | 30.7% |
    ///
    /// A shorter wait cuts a sentence at every hesitation, and the accurate
    /// lane then gets short, badly-bounded utterances -- which is the condition
    /// measurements caught whisper condensing and repeating itself in.
    /// 0.25 s buys 377 ms of translation latency for **twice** the word error
    /// rate, and only 0.60 keeps G2a's ≤15%. So the default stays where it is,
    /// and this is a knob for someone who has read the table.
    #[serde(default = "endpoint_silence_s")]
    pub endpoint_silence_s: f32,
    /// Seconds after which the fast lane closes a line whatever the speaker is
    /// doing. Read from `[asr.fast]`, like `endpoint_silence_s`.
    ///
    /// The two are not alternatives. `endpoint_silence_s` finds the sentence;
    /// this one admits there was no finding it, and lands wherever the clock
    /// happens to be -- usually mid-clause. So it is only ever reached by a
    /// speaker going fast enough that their breaths fall short of 0.6 s, and
    /// the value is a bound on how long the reader waits in that case. See
    /// [`li_asr::DEFAULT_MAX_UTTERANCE_S`] for the sweep behind the 12.
    #[serde(default = "max_utterance_s")]
    pub max_utterance_s: f32,
    /// Restore punctuation and casing on this lane's text.
    /// Read from `[asr.fast]`; the accurate lane's model writes its own.
    ///
    /// The fast lane emits neither, and that costs more than looks: it was measured that
    /// measured that `li_mt::chunk::split` decides whether to cut a line by
    /// whether the line has any mark at all, so an unpunctuated 40-word line
    /// went into NLLB whole and came back as its first clause. Real marks turn
    /// that blind width cut off by themselves. The casing replaces the earlier
    /// `readable()` stopgap, which could only capitalise the first letter
    /// because there was no other information to go on.
    ///
    /// English only, like the fast lane's own model, and **optional**: if the
    /// model is not in the cache the engine says so once and runs exactly as
    /// it did before. Both ASR lanes fail startup in the same situation; the
    /// asymmetry is deliberate, because this one is an improvement and those
    /// two are the program.
    #[serde(default = "punctuation")]
    pub punctuation: bool,
    /// Which punctuation model, by the id `li_core::models` resolves. Read
    /// from `[asr.fast]`, and only when `punctuation` is on.
    #[serde(default = "punct_model")]
    pub punct_model: String,
    /// End a line where the restored punctuation says a sentence ended, instead
    /// of waiting for the endpoint detector.
    ///
    /// **Default off.** This is the one part of punctuation restoration that moves a line
    /// *boundary*, and moving a boundary is not free: the accurate lane answers
    /// one whole utterance at a time, so its single answer then has to be
    /// sliced across the lines by timestamp, which `li_stream` measured at
    /// 19.1% content WER against 18.7% uncut when the slicing was done at a
    /// 12-word cap. The argument for trying anyway is that a sentence onset is
    /// a far better-conditioned place to slice than a width cap -- nobody is
    /// speaking across it -- but that is a hypothesis, and the switch stays off
    /// until it is measured on the user's own audio.
    ///
    /// What it buys, measured: where speech runs on without pauses,
    /// a restored full stop is stable a median 3.52 s (p90 6.72 s) before the
    /// line it sits in closes. Where the endpoint detector is already cutting
    /// well there are no mid-line boundaries at all, so this does nothing --
    /// `ami_meeting` had zero. It fires on the audio the user complained about
    /// and is silent the rest of the time.
    ///
    /// Needs `punctuation`: with no marks there is nothing to cut on, so the
    /// two never need to be kept in step.
    #[serde(default)]
    pub semantic_cut: bool,
}

fn endpoint_silence_s() -> f32 {
    li_asr::DEFAULT_ENDPOINT_SILENCE_S
}

fn max_utterance_s() -> f32 {
    li_asr::DEFAULT_MAX_UTTERANCE_S
}

fn punctuation() -> bool {
    true
}

fn punct_model() -> String {
    "online-punct-en-2024-08-06".into()
}

impl EngineConfig {
    /// The line-breaking and promotion rules, from the file rather than from
    /// [`StreamConfig::default`].
    ///
    /// Originally the engine built `StreamConfig::default()` and nothing
    /// else, so `[asr.accurate] promote_after_s` parsed, validated, round-
    /// tripped through the settings window -- and did nothing at all. A setting
    /// that is read but never applied is worse than one that does not exist.
    pub fn stream(&self) -> li_stream::StreamConfig {
        let d = li_stream::StreamConfig::default();
        li_stream::StreamConfig {
            max_words: self.asr.max_line_words.max(1),
            // The accurate lane is the one being waited for; the fast lane is
            // what gets promoted when the wait runs out.
            promote_after_s: self
                .asr
                .accurate
                .as_ref()
                .map_or(d.promote_after_s, |a| a.promote_after_s),
            // One number, two places that must agree: `li-stream` decides when
            // a line is finished, sherpa decides when a sentence is. They are
            // the same 0.6 s, and letting them drift would put a
            // line boundary where no endpoint is.
            pause_flush_s: self
                .asr
                .fast
                .as_ref()
                .map_or(d.pause_flush_s, |f| f.endpoint_silence_s as f64),
            ..d
        }
    }
}

fn promote_after_s() -> f64 {
    li_stream::StreamConfig::default().promote_after_s
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MtCfg {
    pub backend: String,
    pub model: String,
    /// A second translator, asked only for a piece NLLB could not write a
    /// character of (`li_mt::opus`). Optional the way the punctuation model is:
    /// not in the cache, and the gap is marked □ instead. Empty turns it off.
    pub fallback_model: String,
    pub target: String,
    /// Only a cloud/LLM backend can use these; the local NLLB is measurably
    /// harmed by them and ignores the setting.
    pub context_sentences: usize,
    /// OpenCC config, applied when the backend does not promise Taiwan usage.
    pub opencc: String,
    /// The four settings that were measured and locked.
    pub beam_size: usize,
    /// Cut a line into clauses above this many words before translating.
    /// `0` disables splitting, which is how a third of the lines end up
    /// stopping at a comma.
    pub max_chunk_words: usize,
    /// Cut a stretch with no punctuation at all into equal parts of at most
    /// this many words. `0` leaves it whole, which is what the fast lane --
    /// which punctuates nothing -- was getting: 40 words in, one clause back.
    pub max_run_words: usize,
    pub trim_final_stop: bool,
    /// `0` means 4, which is what measured fastest -- 8 is slower, because the
    /// P and E cores end up oversubscribed.
    pub threads: usize,
    /// Translate the fast lane's sentence without waiting for the accurate one,
    /// and start translating it before the sentence has even closed.
    ///
    /// Two measured steps, on `ami_meeting.wav`. Drafting from the fast lane's
    /// endpoint put the Chinese on the bar at 1052 ms instead of
    /// 1468. Then: the fast lane stops changing its mind 631-647 ms
    /// before the line closes -- the words are done, and what is left is
    /// `endpoint_silence_s` counting out the quiet -- so NLLB is handed the
    /// sentence during that silence rather than after it. The draft now lands
    /// at **676 ms, the same instant as the English**, and the model pass is
    /// invisible.
    ///
    /// It is not more work. Guesses at mid-sentence pauses are thrown away
    /// (12 per 100 s), but every draft they do serve costs nothing, and total
    /// MT time over the clip was unchanged at 15.7 s.
    ///
    /// The cost is that the translation is replaced once, from text with no
    /// punctuation and a higher error rate. Turn this off to go back to one
    /// translation per line, arriving when the line is settled.
    pub draft_from_fast_lane: bool,
    /// Translate a sentence that has ended without waiting for the line to.
    ///
    /// `draft_from_fast_lane` is early by the length of one endpoint: it hands
    /// NLLB the sentence while `endpoint_silence_s` counts out the quiet. That
    /// only helps a speaker who pauses. It was measured what happens to
    /// one who does not: on a clip at 1.5x speed the fast lane's own restored
    /// full stop is stable a **median 3.52 s, p90 6.72 s** before the line it
    /// sits in closes, and on the user's own recording 1.92 s. That wait is the
    /// complaint this whole task came from -- "講太快時來不及顯示翻譯".
    ///
    /// So when a full stop inside the open line has survived two consecutive
    /// hypotheses and has a word decoded after it, the prefix up to it is
    /// translated and shown as a draft. The line is **not** cut: segmentation,
    /// the accurate lane's window, and every deterministic number in the eval
    /// harness are untouched, which is what separates this from stage S2.
    ///
    /// What it costs, and why it is bounded. The one MT thread cannot be
    /// pre-empted once a pass starts, so a mid-line draft can delay the settled
    /// translation by that pass -- the same bound `draft_from_fast_lane`
    /// already carries, and for the same reason `drop_superseded_drafts` keeps
    /// only the newest draft still waiting. The Chinese row is also replaced
    /// once more per line than before.
    ///
    /// Needs `draft_from_fast_lane` (it is the same draft path) and
    /// `[asr.fast] punctuation` (without restored marks there is no boundary to
    /// find). With either off it simply never fires.
    pub draft_mid_line: bool,
}

impl Default for AudioCfg {
    fn default() -> Self {
        Self {
            source: DeviceSelector::SystemLoopback,
        }
    }
}

impl Default for AsrCfg {
    fn default() -> Self {
        Self {
            language: "en".into(),
            fast: Some(LaneCfg {
                backend: "sherpa".into(),
                model: "streaming-zipformer-en-2023-06-21".into(),
                device: DeviceRequest::Auto,
                threads: 0,
                promote_after_s: promote_after_s(),
                endpoint_silence_s: endpoint_silence_s(),
                max_utterance_s: max_utterance_s(),
                punctuation: punctuation(),
                punct_model: punct_model(),
                semantic_cut: false,
            }),
            accurate: Some(LaneCfg {
                backend: "whispercpp".into(),
                model: "small.en-q5_1".into(),
                device: DeviceRequest::Auto,
                threads: 0,
                promote_after_s: promote_after_s(),
                endpoint_silence_s: endpoint_silence_s(),
                max_utterance_s: max_utterance_s(),
                punctuation: punctuation(),
                punct_model: punct_model(),
                semantic_cut: false,
            }),
            max_line_words: max_line_words(),
        }
    }
}

impl Default for MtCfg {
    fn default() -> Self {
        Self {
            backend: "local".into(),
            model: "nllb-200-distilled-600m-ct2-int8".into(),
            fallback_model: "opus-mt-en-zh-ct2-int8".into(),
            target: "zh-Hant-TW".into(),
            context_sentences: 2,
            opencc: "s2twp".into(),
            beam_size: 4,
            max_chunk_words: 6,
            max_run_words: 6,
            trim_final_stop: false,
            threads: 0,
            draft_from_fast_lane: true,
            draft_mid_line: true,
        }
    }
}

impl EngineConfig {
    /// `$LI_CONFIG`, else the platform's own settings directory
    /// (`li_types::paths`).
    pub fn path() -> PathBuf {
        if let Some(p) = std::env::var_os("LI_CONFIG").filter(|s| !s.is_empty()) {
            return PathBuf::from(p);
        }
        li_types::paths::config_file()
    }

    /// Read the config file, or hand back the defaults if there is not one.
    ///
    /// No file is the normal first run, not a failure. A file that will not
    /// parse *is* a failure, reported with the file name and the line -- the
    /// alternative is running for an hour on settings the user thought they had
    /// changed.
    pub fn load() -> Result<Self> {
        Self::load_from(&Self::path())
    }

    pub fn load_from(path: &std::path::Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                toml::from_str(&text).with_context(|| format!("reading {}", path.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    pub fn save(&self) -> Result<PathBuf> {
        let path = Self::path();
        self.save_to(&path)?;
        Ok(path)
    }

    /// Write the settings back, keeping whatever else the file said.
    ///
    /// TOML was chosen so the file would be readable and editable by hand,
    /// and the settings window now writes the same file. A writer
    /// that serialised the struct and truncated would delete every comment in
    /// it the first time someone moved a slider, and would delete the
    /// `[asr.cloud]` key somebody filled in ahead of time -- this program
    /// does not read that section yet, which is not a reason to destroy it.
    ///
    /// So the existing document is edited in place: a value that has not
    /// changed is not rewritten at all, and a section this struct knows nothing
    /// about is left alone. The cost is that nothing is ever *removed* -- a key
    /// that stops being meaningful stays until someone deletes it.
    pub fn save_to(&self, path: &std::path::Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let existing = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let mut doc: toml_edit::DocumentMut = existing
            .parse()
            .with_context(|| format!("reading {}", path.display()))?;
        let fresh: toml_edit::DocumentMut = toml::to_string(self)
            .context("serialising the config")?
            .parse()
            .context("serialising the config")?;
        merge(doc.as_table_mut(), fresh.as_table());

        // Through a temp file and a rename: a settings window that crashes
        // mid-write must not leave a config that will not parse.
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, doc.to_string())
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
    }
}

/// Copy `src` over `dest`, touching only what differs.
///
/// Assignment through the existing slot rather than `insert`, so the key keeps
/// its position in the file and the comment written above it.
fn merge(dest: &mut toml_edit::Table, src: &toml_edit::Table) {
    for (key, item) in src.iter() {
        match (dest.get_mut(key), item) {
            (Some(toml_edit::Item::Table(d)), toml_edit::Item::Table(s)) => merge(d, s),
            (Some(slot), _) => {
                if slot.to_string().trim() != item.to_string().trim() {
                    *slot = item.clone();
                }
            }
            (None, _) => {
                dest.insert(key, item.clone());
            }
        }
    }
}

impl LaneCfg {
    /// Resolve this lane into something `li_asr::build` can open.
    pub fn to_spec(&self, models: &Models, kind: Kind, language: &str) -> Result<LaneSpec> {
        Ok(LaneSpec {
            backend: self.backend.clone(),
            model: models.resolve(kind, &self.model)?,
            device: self.device,
            threads: if self.threads > 0 {
                self.threads
            } else {
                default_threads(kind)
            },
            language: language.to_owned(),
            endpoint_silence_s: self.endpoint_silence_s,
            max_utterance_s: self.max_utterance_s,
        })
    }
}

/// What each lane was measured with.
///
/// The fast lane is a small streaming model and gains nothing above two
/// threads; the accurate one gets the rest of the machine minus two, so the
/// capture callback and the fast lane still have somewhere to run.
fn default_threads(kind: Kind) -> usize {
    match kind {
        Kind::Fast => 2,
        _ => std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(2).max(1))
            .unwrap_or(4),
    }
}

impl MtCfg {
    pub fn to_nllb(&self, models: &Models) -> Result<li_mt::NllbConfig> {
        Ok(li_mt::NllbConfig {
            model_dir: models.resolve(Kind::Mt, &self.model)?,
            threads: if self.threads > 0 { self.threads } else { 4 },
            beam_size: self.beam_size,
            max_chunk_words: self.max_chunk_words,
            max_run_words: self.max_run_words,
            trim_final_stop: self.trim_final_stop,
            fallback_model_dir: self.fallback_dir(models),
            ..Default::default()
        })
    }

    fn fallback_dir(&self, models: &Models) -> Option<PathBuf> {
        if self.fallback_model.is_empty() {
            return None;
        }
        match models.resolve(Kind::Mt, &self.fallback_model) {
            Ok(dir) => Some(dir),
            Err(e) => {
                tracing::info!("translation fallback: off ({e:#})");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_through_toml() {
        let cfg = EngineConfig::default();
        let back: EngineConfig = toml::from_str(&toml::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(back.asr.accurate.unwrap().model, "small.en-q5_1");
        assert_eq!(back.mt.opencc, "s2twp");
    }

    #[test]
    fn an_unknown_device_is_rejected_when_the_file_is_read() {
        // The failure this prevents: `device = "vulcan"` starting on the CPU
        // and nobody noticing until someone wonders why it is slow.
        let err = toml::from_str::<EngineConfig>(
            "[asr.accurate]\nbackend = \"whispercpp\"\nmodel = \"small.en-q5_1\"\ndevice = \"vulcan\"\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("vulcan"), "{err}");
    }

    #[test]
    fn the_transcript_section_is_the_writers_own_config() {
        // `li-transcript` owns it, so a format that does not exist is a startup
        // error here rather than a file that never appears.
        let cfg: EngineConfig =
            toml::from_str("[transcript]\nformats = [\"srt\", \"vtt\"]\n").unwrap();
        assert_eq!(
            cfg.transcript.formats,
            vec![li_transcript::Format::Srt, li_transcript::Format::Vtt]
        );
        let err = toml::from_str::<EngineConfig>("[transcript]\nformats = [\"sbt\"]\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("sbt"), "{err}");
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("li-core-config").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_missing_config_file_means_the_defaults_not_an_error() {
        // First run is the common case, not a failure.
        let path = scratch("absent").join("nested/config.toml");
        let cfg = EngineConfig::load_from(&path).unwrap();
        assert_eq!(cfg.mt.beam_size, 4);
    }

    #[test]
    fn a_malformed_config_file_is_refused_with_its_name() {
        // What this prevents: running for an hour on settings the user thinks
        // they changed.
        let path = scratch("broken").join("config.toml");
        std::fs::write(&path, "[mt]\nbeam_size = \"four\"\n").unwrap();
        let err = format!("{:#}", EngineConfig::load_from(&path).unwrap_err());
        assert!(err.contains("config.toml"), "{err}");
    }

    #[test]
    fn a_saved_config_reads_back_the_same() {
        let path = scratch("roundtrip").join("config.toml");
        let mut cfg = EngineConfig::default();
        cfg.mt.beam_size = 2;
        cfg.transcript.bilingual_file = true;
        cfg.save_to(&path).unwrap();
        let back = EngineConfig::load_from(&path).unwrap();
        assert_eq!(back.mt.beam_size, 2);
        assert!(back.transcript.bilingual_file);
        assert!(
            !path.with_extension("toml.tmp").exists(),
            "the temp file is renamed away"
        );
    }

    #[test]
    fn saving_keeps_the_comments_and_the_sections_this_program_does_not_read() {
        // What this prevents: someone writes their notes and their Deepgram key
        // into config.toml (the schema reserves `[asr.cloud]`, which nothing has
        // not built yet), then moves the opacity slider, and both are gone.
        let path = scratch("handwritten").join("config.toml");
        std::fs::write(
            &path,
            "# the quiet room, measured 2026-09-03\n[ui]\n# 22 is too small on the 4K panel\nfont_size = 28.0\n\n[asr.cloud]\ndeepgram_key = \"kept\"\n",
        )
        .unwrap();

        let mut cfg = EngineConfig::load_from(&path).unwrap();
        assert_eq!(cfg.ui.font_size, 28.0, "the file wins over the default");
        cfg.ui.opacity = 0.5;
        cfg.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# the quiet room"), "{text}");
        assert!(text.contains("# 22 is too small"), "{text}");
        assert!(text.contains("deepgram_key = \"kept\""), "{text}");
        assert!(text.contains("font_size = 28.0"), "{text}");
        assert_eq!(EngineConfig::load_from(&path).unwrap().ui.opacity, 0.5);
    }

    #[test]
    fn the_audio_source_is_one_string_and_leaves_nothing_behind() {
        // A tagged table would write `[audio.source] kind = "device", id = ...`
        // and the merge above never deletes, so switching back to the system
        // mixer would leave the old device id sitting under it.
        let path = scratch("source").join("config.toml");
        let mut cfg = EngineConfig::default();
        cfg.audio.source = DeviceSelector::Device("alsa_output.pci.monitor".into());
        cfg.save_to(&path).unwrap();
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("source = \"alsa_output.pci.monitor\""),
        );
        cfg.audio.source = DeviceSelector::Microphone;
        cfg.save_to(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("source = \"mic\""), "{text}");
        assert!(!text.contains("alsa_output"), "{text}");
        assert_eq!(
            EngineConfig::load_from(&path).unwrap().audio.source,
            DeviceSelector::Microphone
        );
    }

    #[test]
    fn a_lane_resolves_its_model_and_fills_in_the_measured_thread_count() {
        let root = scratch("lane");
        std::fs::create_dir_all(root.join("sherpa-onnx-zip")).unwrap();
        let models = Models::with_root(&root);
        let lane = LaneCfg {
            backend: "sherpa".into(),
            model: "zip".into(),
            device: DeviceRequest::Auto,
            threads: 0,
            promote_after_s: 8.0,
            endpoint_silence_s: 0.35,
            max_utterance_s: 9.0,
            punctuation: true,
            punct_model: "online-punct-en-2024-08-06".into(),
            semantic_cut: false,
        };
        let spec = lane.to_spec(&models, Kind::Fast, "en").unwrap();
        assert_eq!(spec.model, root.join("sherpa-onnx-zip"));
        assert_eq!(spec.threads, 2, "the fast lane gains nothing above two");
        assert_eq!(spec.language, "en");
        // The one setting that decides how late a finished sentence is, so it
        // has to survive the trip from the file into the backend.
        assert_eq!(spec.endpoint_silence_s, 0.35);
        // Same trip, same reason: a cap that stops at the config layer leaves
        // the fast lane running sherpa's own 20 s and the file lying about it.
        assert_eq!(spec.max_utterance_s, 9.0);

        // An explicit setting wins over the measured default.
        let lane = LaneCfg { threads: 7, ..lane };
        assert_eq!(lane.to_spec(&models, Kind::Fast, "en").unwrap().threads, 7);
    }

    #[test]
    fn the_mt_section_carries_the_settings_task_1_6_locked() {
        let root = scratch("mt");
        std::fs::create_dir_all(root.join("nllb")).unwrap();
        let models = Models::with_root(&root);
        let cfg = MtCfg {
            model: "nllb".into(),
            ..MtCfg::default()
        };
        let nllb = cfg.to_nllb(&models).unwrap();
        assert_eq!(nllb.model_dir, root.join("nllb"));
        assert_eq!(nllb.beam_size, 4);
        assert_eq!(
            nllb.max_chunk_words, 6,
            "0 would put a third of lines back on a comma"
        );
        assert_eq!(
            nllb.max_run_words, 6,
            "0 would leave the fast lane's unpunctuated 40-word lines whole"
        );
        assert!(!nllb.trim_final_stop);
        assert_eq!(nllb.threads, 4, "8 is slower: P/E core oversubscription");
    }

    #[test]
    fn the_whole_config_survives_a_trip_through_json() {
        // The settings window edits it as JSON: it reads this struct, changes
        // a field, and hands the whole thing back to be written. A field that
        // does not survive that trip is a save that fails at run time with a
        // serde message, which is not the kind of thing to find out about from
        // a user.
        let mut cfg = EngineConfig::default();
        cfg.audio.source = DeviceSelector::Device("alsa_input.usb".into());
        cfg.ui.position = BarPosition::Top;
        cfg.asr.accurate = None; // the no-GPU case: a lane that is not there
        let back: EngineConfig = serde_json::from_str(&serde_json::to_string(&cfg).unwrap())
            .expect("the settings window's round trip");
        assert_eq!(back.audio.source, cfg.audio.source);
        assert_eq!(back.ui, cfg.ui);
        assert_eq!(back.hotkeys, cfg.hotkeys);
        assert!(back.asr.accurate.is_none());
        assert_eq!(back.transcript.formats, cfg.transcript.formats);
    }

    #[test]
    fn the_utterance_cap_comes_from_the_file_and_defaults_to_the_measured_twelve() {
        let cfg: EngineConfig = toml::from_str(
            r#"
            [asr.fast]
            backend = "sherpa"
            model = "streaming-zipformer-en-2023-06-21"
            max_utterance_s = 7.5
            "#,
        )
        .unwrap();
        let fast = cfg.asr.fast.as_ref().expect("the file names a fast lane");
        assert_eq!(fast.max_utterance_s, 7.5);
        // The default is the measured number, not sherpa's own 20:
        // 20 is where a 1.5x-speed clip spent a fifth of itself in one line
        // that the accurate lane then timed out on.
        let d: EngineConfig =
            toml::from_str("[asr.fast]\nbackend = \"sherpa\"\nmodel = \"m\"\n").unwrap();
        assert_eq!(d.asr.fast.unwrap().max_utterance_s, 12.0);
    }

    #[test]
    fn a_partial_file_keeps_the_defaults() {
        let cfg: EngineConfig = toml::from_str("[mt]\nbackend = \"gemini\"\n").unwrap();
        assert_eq!(cfg.mt.backend, "gemini");
        assert_eq!(cfg.mt.target, "zh-Hant-TW");
        assert!(cfg.asr.fast.is_some());
    }

    #[test]
    fn the_line_breaking_rules_come_from_the_file_now() {
        // All three of these parsed and were then thrown away: the engine built
        // `StreamConfig::default()`. `promote_after_s` had been round-tripping
        // through the settings window doing nothing since the engine was first assembled.
        let cfg: EngineConfig = toml::from_str(
            r#"
            [asr]
            max_line_words = 12
            [asr.fast]
            backend = "sherpa"
            model = "streaming-zipformer-en-2023-06-21"
            endpoint_silence_s = 0.45
            [asr.accurate]
            backend = "whispercpp"
            model = "small.en-q5_1"
            promote_after_s = 3.5
            "#,
        )
        .unwrap();
        let s = cfg.stream();
        assert_eq!(s.max_words, 12);
        assert_eq!(s.promote_after_s, 3.5);
        // Not `assert_eq!`: `endpoint_silence_s` is an `f32` because that is
        // what sherpa takes, and widening it back to `f64` leaves a tail.
        assert!(
            (s.pause_flush_s - 0.45).abs() < 1e-6,
            "one endpoint, not two"
        );
    }

    #[test]
    fn a_file_that_says_nothing_still_gets_the_measured_defaults() {
        let s = EngineConfig::default().stream();
        let d = li_stream::StreamConfig::default();
        assert_eq!(
            (s.max_words, s.promote_after_s),
            (d.max_words, d.promote_after_s)
        );
        assert!((s.pause_flush_s - d.pause_flush_s).abs() < 1e-6);
    }

    #[test]
    fn a_line_may_not_be_zero_words_long() {
        let cfg: EngineConfig = toml::from_str("[asr]\nmax_line_words = 0\n").unwrap();
        assert_eq!(
            cfg.stream().max_words,
            1,
            "0 would divide a line by nothing"
        );
    }
}
