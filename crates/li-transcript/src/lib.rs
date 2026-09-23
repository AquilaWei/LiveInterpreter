//! Transcript writers.
//!
//! The pre-translation source transcript is a hard requirement of this project,
//! and it goes to its **own file**, separate from any bilingual output: it is
//! what gets proofread, searched and fed to other tools afterwards, and mixing
//! a translation into it destroys all three uses.
//!
//! Lines are appended as they are finalised rather than written at the end, so
//! killing the process keeps everything already recognised. That is a claim
//! about behaviour, so `tests/crash.rs` makes it one: a child process writes a
//! session, is stopped with SIGKILL mid-sentence, and the files are read back.
//! Two implementation decisions follow from it and are easy to undo by
//! accident, so both are stated where they live -- no `BufWriter` anywhere
//! ([`Writer`]), and the `.jsonl` is an append-only log rather than a file that
//! gets rewritten when translations arrive ([`jsonl`]).
//!
//! ## What a session writes
//!
//! One set of files per session, sharing a timestamped stem, in `cfg.dir`:
//!
//! | file | contents |
//! |---|---|
//! | `<stem>_en.txt` | the source transcript, one sentence per line |
//! | `<stem>_en.srt` / `.vtt` | the same text with subtitle timings |
//! | `<stem>_en.jsonl` | `{line_id, start, end, source, translation, lane, reason}` |
//! | `<stem>_en-zh.txt` | optional; source and translation interleaved |
//!
//! Only the `.jsonl` and the bilingual file carry translations, and the
//! bilingual file is off by default.
//!
//! **The text is the accurate lane's**. The fast lane has no
//! punctuation or capitalisation and a materially higher error rate; it reaches
//! the screen and nothing else, unless the 8 s promotion rule fires -- and then
//! `lane` and `reason` in the `.jsonl` say so, which is the whole reason those
//! two fields exist.

use anyhow::Result;
use li_types::TranscriptLine;
use serde::{Deserialize, Serialize};

pub mod jsonl;
pub mod render;
mod writer;

pub use writer::{Writer, expand_home};

/// An output format. Typed rather than a string so `formats = ["sbt"]` is a
/// startup error naming the four that exist, not a file that never appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Txt,
    Srt,
    Vtt,
    Jsonl,
}

impl Format {
    pub fn ext(self) -> &'static str {
        match self {
            Format::Txt => "txt",
            Format::Srt => "srt",
            Format::Vtt => "vtt",
            Format::Jsonl => "jsonl",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.ext())
    }
}

/// `[transcript]` in `config.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TranscriptConfig {
    pub enabled: bool,
    /// A leading `~` is expanded; the directory is created if it is missing.
    pub dir: std::path::PathBuf,
    pub formats: Vec<Format>,
    /// The source transcript always gets its own file; this adds a
    /// second, bilingual one beside it.
    pub bilingual_file: bool,
}

impl Default for TranscriptConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: "~/Documents/LiveInterpreter".into(),
            formats: vec![Format::Txt, Format::Jsonl],
            bilingual_file: false,
        }
    }
}

/// The two language tags that end up in the filenames.
///
/// A struct rather than two `&str` parameters because they are the same type
/// and swapping them produces a plausible-looking wrong filename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Langs {
    pub source: String,
    pub target: String,
}

impl Default for Langs {
    fn default() -> Self {
        Self {
            source: "en".into(),
            target: "zh".into(),
        }
    }
}

/// Where finalised lines go.
///
/// A trait because Android writes through the Storage Access Framework rather
/// than to a path, and because `li-core`'s tests need a sink that keeps
/// its lines in memory.
pub trait TranscriptSink: Send {
    /// Accurate-lane text, or fast-lane text that was promoted. Called once per
    /// `line_id`: `li-stream` emits `SourceFinal` in order and never revises it.
    fn on_source_final(&mut self, line: &TranscriptLine) -> Result<()>;

    /// Fills in the translation for a line already written.
    fn on_translation_final(&mut self, line_id: u64, text: &str) -> Result<()>;

    /// Ends the session and tidies what only a clean shutdown can tidy.
    ///
    /// There is no `flush`: every append is already durable against the process
    /// dying, which is the point of the whole crate.
    fn finish(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_defaults_match_plan_14() {
        let cfg = TranscriptConfig::default();
        assert!(cfg.enabled && !cfg.bilingual_file);
        assert_eq!(cfg.formats, vec![Format::Txt, Format::Jsonl]);
        assert_eq!(
            cfg.dir,
            std::path::PathBuf::from("~/Documents/LiveInterpreter")
        );
    }

    #[test]
    fn a_misspelled_format_is_rejected_with_the_list_of_real_ones() {
        // What this prevents: `formats = ["sbt"]` starting fine and the user
        // wondering for an hour why no subtitle file appears.
        let err = toml::from_str::<TranscriptConfig>("formats = [\"sbt\"]\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("sbt") && err.contains("srt"), "{err}");
    }

    #[test]
    fn a_partial_section_keeps_the_defaults() {
        let cfg: TranscriptConfig = toml::from_str("bilingual_file = true\n").unwrap();
        assert!(cfg.bilingual_file);
        assert_eq!(cfg.formats, vec![Format::Txt, Format::Jsonl]);
    }
}
