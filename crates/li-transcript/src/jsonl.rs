//! The `.jsonl` file: one JSON object per line, append-only.
//!
//! ## Why a line can appear twice
//!
//! A line's source text is final the moment the accurate lane commits it; its
//! translation arrives a few hundred milliseconds later (measured
//! 195-368 ms), by which time the source is already on disk. The two ways out
//! of that are to hold the record back until the translation lands, or to write
//! what is known and patch it afterwards.
//!
//! Holding it back loses recognised speech on a crash, which is the one thing
//! this crate must not do. So the file is a log: the source
//! record goes down immediately, and a translation appends a second record
//! carrying the same `line_id` and nothing but the new field. **A reader folds
//! the file by `line_id`, last value wins** -- which is what [`read`] does.
//!
//! [`crate::Writer::finish`] collapses the log back into one record per line on
//! a clean shutdown, so the patch form is what you find after a crash and not
//! what you find normally. Both parse the same way.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use li_types::{FastReason, Lane, TranscriptLine};
use serde::{Deserialize, Serialize};

/// One record as it appears in the file.
///
/// Every field but `line_id` is optional, because a patch record carries only
/// what changed. Field names are `start`/`end`, in seconds,
/// rather than the Rust field names, which carry their unit instead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub line_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<f64>,
    /// The pre-translation source text. An early draft called this field `source`
    /// and also used `"source": "fast"` for the lane; the lane is `lane` here,
    /// as [`TranscriptLine`] has it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<Lane>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<FastReason>,
}

impl Record {
    /// The record written when a line is finalised.
    pub fn source(line: &TranscriptLine) -> Self {
        Self {
            line_id: line.line_id,
            // Milliseconds are the resolution of every other file here, and
            // rounding keeps `0.30000000000000004` out of the transcript.
            start: Some(round_ms(line.start_s)),
            end: Some(round_ms(line.end_s)),
            source: Some(line.source.clone()),
            translation: line.translation.clone(),
            lane: Some(line.lane),
            reason: line.reason,
        }
    }

    /// The record written when the translation for an already-written line
    /// arrives.
    pub fn patch_translation(line_id: u64, text: &str) -> Self {
        Self {
            line_id,
            translation: Some(text.to_owned()),
            ..Self::default()
        }
    }

    /// Apply a later record over an earlier one. A field the later record does
    /// not carry is left alone -- that is what makes a patch a patch.
    pub fn merge(&mut self, other: Record) {
        let Record {
            line_id: _,
            start,
            end,
            source,
            translation,
            lane,
            reason,
        } = other;
        self.start = start.or(self.start);
        self.end = end.or(self.end);
        self.source = source.or_else(|| self.source.take());
        self.translation = translation.or_else(|| self.translation.take());
        self.lane = lane.or(self.lane);
        self.reason = reason.or(self.reason);
    }

    pub fn to_line(&self) -> String {
        let mut s = serde_json::to_string(self).expect("a Record is always serialisable");
        s.push('\n');
        s
    }
}

/// Read a `.jsonl` transcript back, folding patch records into their lines.
///
/// Returns the lines in `line_id` order. Used by [`crate::Writer::finish`] to
/// collapse the log, and by anything reading a transcript afterwards.
pub fn read(path: &Path) -> Result<Vec<Record>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut lines: BTreeMap<u64, Record> = BTreeMap::new();
    for (i, text) in BufReader::new(file).lines().enumerate() {
        let text = text.with_context(|| format!("read {}", path.display()))?;
        if text.trim().is_empty() {
            continue;
        }
        let rec: Record = serde_json::from_str(&text)
            .with_context(|| format!("{}:{}: not a transcript record", path.display(), i + 1))?;
        match lines.get_mut(&rec.line_id) {
            Some(existing) => existing.merge(rec),
            None => {
                lines.insert(rec.line_id, rec);
            }
        }
    }
    Ok(lines.into_values().collect())
}

fn round_ms(v: f64) -> f64 {
    if v.is_finite() {
        (v * 1000.0).round() / 1000.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit tests do not get `CARGO_TARGET_TMPDIR` -- that is set for
    /// integration tests only.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("li-transcript-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{name}.jsonl"))
    }

    fn line(line_id: u64, source: &str) -> TranscriptLine {
        TranscriptLine {
            line_id,
            start_s: 0.1 + 0.2,
            end_s: 2.0,
            source: source.into(),
            translation: None,
            lane: Lane::Accurate,
            reason: None,
        }
    }

    #[test]
    fn a_source_record_carries_the_fields_plan_2_6_names() {
        let s = Record::source(&line(3, "Hello.")).to_line();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["line_id"], 3);
        assert_eq!(
            v["start"], 0.3,
            "floating point noise reaches the file: {s}"
        );
        assert_eq!(v["end"], 2.0);
        assert_eq!(v["source"], "Hello.");
        assert_eq!(v["lane"], "accurate");
        assert!(v.get("translation").is_none(), "{s}");
        assert!(v.get("reason").is_none(), "{s}");
    }

    #[test]
    fn a_patch_carries_only_what_changed() {
        let s = Record::patch_translation(3, "您好。").to_line();
        assert_eq!(s, "{\"line_id\":3,\"translation\":\"您好。\"}\n");
    }

    #[test]
    fn a_patch_fills_in_the_translation_without_touching_the_source() {
        let mut rec = Record::source(&line(3, "Hello."));
        rec.merge(Record::patch_translation(3, "您好。"));
        assert_eq!(rec.source.as_deref(), Some("Hello."));
        assert_eq!(rec.translation.as_deref(), Some("您好。"));
        assert_eq!(rec.lane, Some(Lane::Accurate));
    }

    #[test]
    fn reading_folds_the_log_and_orders_by_line_id() {
        let path = scratch("jsonl_fold");
        let body = [
            Record::source(&line(1, "One.")).to_line(),
            Record::source(&line(2, "Two.")).to_line(),
            Record::patch_translation(1, "一。").to_line(),
            Record::patch_translation(2, "二。").to_line(),
        ]
        .concat();
        std::fs::write(&path, body).unwrap();

        let got = read(&path).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].line_id, 1);
        assert_eq!(got[0].source.as_deref(), Some("One."));
        assert_eq!(got[0].translation.as_deref(), Some("一。"));
        assert_eq!(got[1].translation.as_deref(), Some("二。"));
    }

    #[test]
    fn a_corrupt_line_is_an_error_naming_the_line_number() {
        let path = scratch("jsonl_corrupt");
        std::fs::write(&path, "{\"line_id\":1}\n{\"line_id\"\n").unwrap();
        let err = format!("{:#}", read(&path).unwrap_err());
        assert!(err.contains(":2:"), "{err}");
    }
}
