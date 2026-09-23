//! The session writer: opens one set of files and appends to them as lines are
//! finalised.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use li_types::TranscriptLine;

use crate::{Format, Langs, TranscriptConfig, TranscriptSink, jsonl, render};

/// A finalised line waiting for its translation before the bilingual file can
/// have it. Only allocated when `bilingual_file` is on.
struct Pending {
    line: TranscriptLine,
    translation: Option<String>,
}

/// Writes one session's transcript files.
///
/// Every append is a single unbuffered `write_all`, so the bytes are in the
/// page cache before the call returns and killing the process cannot lose a
/// line that has already been finalised (`tests/crash.rs`). There is
/// deliberately no `BufWriter` and no `sync_all`: the first would reintroduce
/// exactly the loss this is here to prevent, and the second would fsync the
/// disk once a sentence to defend against power loss, which is not the failure
/// being defended against.
pub struct Writer {
    files: Vec<(Format, File)>,
    paths: Vec<PathBuf>,
    /// Kept separately from `files` because [`Writer::finish`] rewrites it.
    jsonl_path: Option<PathBuf>,
    patched: bool,
    bilingual: Option<File>,
    queue: VecDeque<Pending>,
    translated_through: Option<u64>,
    srt_index: usize,
}

impl Writer {
    /// Create this session's files under `cfg.dir`.
    ///
    /// With `enabled = false` this is a working sink that writes nothing and
    /// creates no directory, so a caller never has to hold an `Option`.
    pub fn open(cfg: &TranscriptConfig, langs: &Langs) -> Result<Self> {
        let mut w = Self {
            files: Vec::new(),
            paths: Vec::new(),
            jsonl_path: None,
            patched: false,
            bilingual: None,
            queue: VecDeque::new(),
            translated_through: None,
            srt_index: 0,
        };
        if !cfg.enabled {
            return Ok(w);
        }

        let dir = expand_home(&cfg.dir);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create transcript directory {}", dir.display()))?;

        let (src, tgt) = (tag(&langs.source), tag(&langs.target));
        let mut formats: Vec<Format> = Vec::new();
        for f in &cfg.formats {
            if !formats.contains(f) {
                formats.push(*f);
            }
        }

        let stem = free_stem(&dir, &formats, &src, &tgt, cfg.bilingual_file)?;
        for f in formats {
            let path = dir.join(format!("{stem}_{src}.{}", f.ext()));
            let mut file = create(&path)?;
            if f == Format::Vtt {
                file.write_all(render::VTT_HEADER.as_bytes())
                    .with_context(|| format!("write {}", path.display()))?;
            }
            if f == Format::Jsonl {
                w.jsonl_path = Some(path.clone());
            }
            w.paths.push(path);
            w.files.push((f, file));
        }
        if cfg.bilingual_file {
            let path = dir.join(format!("{stem}_{src}-{tgt}.txt"));
            w.bilingual = Some(create(&path)?);
            w.paths.push(path);
        }
        Ok(w)
    }

    /// Every file this session is writing, source transcript first. What the
    /// UI's `open_transcript_folder()` points at.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn append(&mut self, format: Format, text: &str) -> Result<()> {
        for (f, file) in &mut self.files {
            if *f == format {
                return file
                    .write_all(text.as_bytes())
                    .with_context(|| format!("append to the {format} transcript"));
            }
        }
        Ok(())
    }

    /// Write out every bilingual pair that can no longer change.
    ///
    /// A pair is ready when its translation has arrived, or when a *later*
    /// line's translation has: translations come back in line order, so one
    /// overtaking means the earlier one is not coming (MT was off, or it
    /// failed). Without the second half of that rule a single missing
    /// translation would hold the whole file back to [`Writer::finish`].
    fn drain_bilingual(&mut self, force: bool) -> Result<()> {
        let Some(file) = self.bilingual.as_mut() else {
            self.queue.clear();
            return Ok(());
        };
        while let Some(front) = self.queue.front() {
            let overtaken = self
                .translated_through
                .is_some_and(|t| front.line.line_id < t);
            if !(force || overtaken || front.translation.is_some()) {
                break;
            }
            let p = self.queue.pop_front().expect("checked above");
            let text = render::bilingual(&p.line, p.translation.as_deref().unwrap_or_default());
            file.write_all(text.as_bytes())
                .context("append to the bilingual transcript")?;
        }
        Ok(())
    }

    /// Rewrite the `.jsonl` log as one record per line (see [`jsonl`]).
    ///
    /// Through a temporary file and a rename, so an interrupted collapse leaves
    /// the previous complete file rather than half of a new one. A log that
    /// cannot be parsed is left exactly as it is: an unreadable line is a line
    /// to hand to a person, not one to drop on their behalf.
    fn collapse_jsonl(&mut self) {
        let (Some(path), true) = (self.jsonl_path.take(), self.patched) else {
            return;
        };
        let body = match jsonl::read(&path) {
            Ok(records) => records
                .iter()
                .map(jsonl::Record::to_line)
                .collect::<String>(),
            Err(e) => {
                tracing::warn!("leaving {} as an append-only log: {e:#}", path.display());
                return;
            }
        };
        let tmp = path.with_extension("jsonl.tmp");
        if let Err(e) = std::fs::write(&tmp, body).and_then(|()| std::fs::rename(&tmp, &path)) {
            tracing::warn!("leaving {} as an append-only log: {e}", path.display());
        }
    }
}

impl TranscriptSink for Writer {
    fn on_source_final(&mut self, line: &TranscriptLine) -> Result<()> {
        if render::one_line(&line.source).is_empty() {
            // `li-stream` keeps the fast lane's text rather than overwriting a
            // line with nothing, so this should not happen -- and
            // if it does, an empty subtitle cue is not the way to find out.
            tracing::debug!(line.line_id, "skipping an empty line");
            return Ok(());
        }
        self.append(Format::Txt, &render::txt(line))?;
        if self.files.iter().any(|(f, _)| *f == Format::Srt) {
            self.srt_index += 1;
            let block = render::srt(self.srt_index, line);
            self.append(Format::Srt, &block)?;
        }
        self.append(Format::Vtt, &render::vtt(line))?;
        self.append(Format::Jsonl, &jsonl::Record::source(line).to_line())?;
        if self.bilingual.is_some() {
            self.queue.push_back(Pending {
                line: line.clone(),
                translation: None,
            });
            self.drain_bilingual(false)?;
        }
        Ok(())
    }

    fn on_translation_final(&mut self, line_id: u64, text: &str) -> Result<()> {
        if self.jsonl_path.is_some() {
            self.patched = true;
            self.append(
                Format::Jsonl,
                &jsonl::Record::patch_translation(line_id, text).to_line(),
            )?;
        }
        if self.bilingual.is_some() {
            self.translated_through = Some(match self.translated_through {
                Some(t) => t.max(line_id),
                None => line_id,
            });
            if let Some(p) = self.queue.iter_mut().find(|p| p.line.line_id == line_id) {
                p.translation = Some(text.to_owned());
            }
            self.drain_bilingual(false)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.drain_bilingual(true)?;
        self.bilingual = None;
        self.collapse_jsonl();
        // Closing the handles makes a later call a no-op, which matters because
        // the collapse has already replaced the `.jsonl` behind them.
        self.files.clear();
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if let Err(e) = self.finish() {
            tracing::warn!("closing the transcript: {e:#}");
        }
    }
}

fn create(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))
}

/// `2026-08-31_193045`, plus `-2`, `-3`... if a session already owns it.
///
/// One stem for the whole session, so a directory listing groups a session's
/// files together and the suffix cannot land on some of them and not others.
fn free_stem(
    dir: &Path,
    formats: &[Format],
    src: &str,
    tgt: &str,
    bilingual: bool,
) -> Result<String> {
    let now = jiff::Zoned::now().strftime("%Y-%m-%d_%H%M%S").to_string();
    for n in 1..1000 {
        let stem = if n == 1 {
            now.clone()
        } else {
            format!("{now}-{n}")
        };
        let mut taken = formats
            .iter()
            .any(|f| dir.join(format!("{stem}_{src}.{}", f.ext())).exists());
        taken |= bilingual && dir.join(format!("{stem}_{src}-{tgt}.txt")).exists();
        if !taken {
            return Ok(stem);
        }
    }
    anyhow::bail!(
        "{} already holds a thousand transcripts from this second",
        dir.display()
    )
}

/// A language tag on its way into a filename. `zh/Hant` would otherwise ask for
/// a directory that does not exist, and an empty tag for a file called `_.txt`.
fn tag(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if out.is_empty() { "x".into() } else { out }
}

/// `~` is a shell convention, not a filesystem one: `create_dir_all` on a path
/// starting with it makes a directory literally named `~`. Public because the
/// file transcriber writes into the same `[transcript] dir`.
pub fn expand_home(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    let rest = s
        .strip_prefix("~/")
        .or_else(|| s.strip_prefix("~\\"))
        .or_else(|| (s == "~").then_some(""));
    match (rest, li_types::paths::home()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => p.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use li_types::Lane;

    fn tmp(name: &str) -> PathBuf {
        // Unit tests do not get `CARGO_TARGET_TMPDIR`; that is for `tests/`.
        let dir = std::env::temp_dir().join("li-transcript-tests").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cfg(dir: &Path, formats: &[Format], bilingual: bool) -> TranscriptConfig {
        TranscriptConfig {
            enabled: true,
            dir: dir.to_path_buf(),
            formats: formats.to_vec(),
            bilingual_file: bilingual,
        }
    }

    pub(crate) fn line(line_id: u64, start_s: f64, end_s: f64, source: &str) -> TranscriptLine {
        TranscriptLine {
            line_id,
            start_s,
            end_s,
            source: source.into(),
            translation: None,
            lane: Lane::Accurate,
            reason: None,
        }
    }

    fn read(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn each_format_gets_its_own_file_named_for_the_session() {
        let dir = tmp("writer_files");
        let all = [Format::Txt, Format::Srt, Format::Vtt, Format::Jsonl];
        let w = Writer::open(&cfg(&dir, &all, true), &Langs::default()).unwrap();
        let names: Vec<String> = w
            .paths()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 5, "{names:?}");
        assert!(names[0].ends_with("_en.txt"), "{names:?}");
        assert!(names[4].ends_with("_en-zh.txt"), "{names:?}");
        // One stem for the whole session.
        let stem = names[0].trim_end_matches("_en.txt").to_owned();
        assert!(names.iter().all(|n| n.starts_with(&stem)), "{names:?}");
    }

    #[test]
    fn the_source_transcript_is_a_file_of_its_own_with_no_translation_in_it() {
        // The hard requirement, in one assertion.
        let dir = tmp("writer_source_only");
        let mut w = Writer::open(&cfg(&dir, &[Format::Txt], true), &Langs::default()).unwrap();
        let src = w.paths()[0].clone();
        let bi = w.paths()[1].clone();
        w.on_source_final(&line(1, 0.0, 2.0, "Hello everybody."))
            .unwrap();
        w.on_translation_final(1, "您好，各位。").unwrap();
        w.finish().unwrap();

        assert_ne!(src, bi);
        assert_eq!(read(&src), "Hello everybody.\n");
        assert!(read(&bi).contains("您好，各位。"));
    }

    #[test]
    fn a_line_is_on_disk_before_its_translation_exists() {
        // The property `tests/crash.rs` then checks against a real SIGKILL.
        let dir = tmp("writer_immediate");
        let mut w = Writer::open(
            &cfg(&dir, &[Format::Txt, Format::Jsonl], false),
            &Langs::default(),
        )
        .unwrap();
        w.on_source_final(&line(1, 0.0, 2.0, "Hello.")).unwrap();
        assert_eq!(read(&w.paths()[0]), "Hello.\n");
        assert!(read(&w.paths()[1]).contains("\"source\":\"Hello.\""));
    }

    #[test]
    fn srt_blocks_are_numbered_from_one_and_vtt_carries_its_header() {
        let dir = tmp("writer_cues");
        let mut w = Writer::open(
            &cfg(&dir, &[Format::Srt, Format::Vtt], false),
            &Langs::default(),
        )
        .unwrap();
        w.on_source_final(&line(1, 0.0, 2.0, "One.")).unwrap();
        w.on_source_final(&line(2, 3.0, 5.0, "Two.")).unwrap();
        w.finish().unwrap();

        assert_eq!(
            read(&w.paths()[0]),
            "1\n00:00:00,000 --> 00:00:02,000\nOne.\n\n2\n00:00:03,000 --> 00:00:05,000\nTwo.\n\n"
        );
        assert!(read(&w.paths()[1]).starts_with("WEBVTT\n\n00:00:00.000 -->"));
    }

    #[test]
    fn finishing_collapses_the_jsonl_log_into_one_record_per_line() {
        let dir = tmp("writer_collapse");
        let mut w = Writer::open(&cfg(&dir, &[Format::Jsonl], false), &Langs::default()).unwrap();
        let path = w.paths()[0].clone();
        w.on_source_final(&line(1, 0.0, 2.0, "One.")).unwrap();
        w.on_source_final(&line(2, 3.0, 5.0, "Two.")).unwrap();
        w.on_translation_final(1, "一。").unwrap();
        w.on_translation_final(2, "二。").unwrap();
        assert_eq!(read(&path).lines().count(), 4, "the log form");

        w.finish().unwrap();
        let body = read(&path);
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].contains("\"source\":\"One.\"") && lines[0].contains("一。"));
        assert!(lines[1].contains("\"source\":\"Two.\"") && lines[1].contains("二。"));
    }

    #[test]
    fn a_missing_translation_does_not_hold_the_bilingual_file_back() {
        // Line 2 never gets one. Line 3's arrival is what proves it never will.
        let dir = tmp("writer_bilingual_gap");
        let mut w = Writer::open(&cfg(&dir, &[Format::Txt], true), &Langs::default()).unwrap();
        let bi = w.paths()[1].clone();
        for (id, t) in [(1, 0.0), (2, 3.0), (3, 6.0)] {
            w.on_source_final(&line(id, t, t + 2.0, &format!("Line {id}.")))
                .unwrap();
        }
        w.on_translation_final(1, "一").unwrap();
        w.on_translation_final(3, "三").unwrap();
        assert_eq!(
            read(&bi),
            "[00:00:00] Line 1.\n           一\n\n\
             [00:00:03] Line 2.\n\n\
             [00:00:06] Line 3.\n           三\n\n",
            "line 2 held line 3 back"
        );
    }

    #[test]
    fn a_disabled_writer_creates_nothing_and_still_accepts_lines() {
        let dir = tmp("writer_disabled").join("not-created");
        let mut c = cfg(&dir, &[Format::Txt], true);
        c.enabled = false;
        let mut w = Writer::open(&c, &Langs::default()).unwrap();
        w.on_source_final(&line(1, 0.0, 2.0, "Hello.")).unwrap();
        w.on_translation_final(1, "您好").unwrap();
        w.finish().unwrap();
        assert!(w.paths().is_empty());
        assert!(!dir.exists());
    }

    #[test]
    fn a_second_session_in_the_same_second_does_not_append_to_the_first() {
        let dir = tmp("writer_collision");
        let c = cfg(&dir, &[Format::Txt], false);
        let a = Writer::open(&c, &Langs::default()).unwrap();
        let b = Writer::open(&c, &Langs::default()).unwrap();
        assert_ne!(a.paths()[0], b.paths()[0]);
    }

    #[test]
    fn an_empty_line_never_reaches_a_subtitle_file() {
        let dir = tmp("writer_empty");
        let mut w = Writer::open(
            &cfg(&dir, &[Format::Txt, Format::Srt], false),
            &Langs::default(),
        )
        .unwrap();
        w.on_source_final(&line(1, 0.0, 2.0, "   ")).unwrap();
        w.on_source_final(&line(2, 3.0, 5.0, "Real.")).unwrap();
        w.finish().unwrap();
        assert_eq!(read(&w.paths()[0]), "Real.\n");
        assert!(
            read(&w.paths()[1]).starts_with("1\n"),
            "the index skips it too"
        );
    }

    #[test]
    fn a_language_tag_cannot_escape_the_transcript_directory() {
        assert_eq!(tag("zh-Hant-TW"), "zh-Hant-TW");
        assert_eq!(tag("zh/Hant"), "zh-Hant");
        assert_eq!(tag("../etc"), "---etc");
        assert_eq!(tag(""), "x");
    }

    #[test]
    fn a_tilde_in_the_configured_directory_means_home() {
        let home = li_types::paths::home().expect("a test host has a home directory");
        assert_eq!(
            expand_home(Path::new("~/Documents")),
            Path::new(&home).join("Documents")
        );
        assert_eq!(expand_home(Path::new("/tmp/x")), Path::new("/tmp/x"));
        // Only a leading `~` -- a directory really called `a~b` stays put.
        assert_eq!(expand_home(Path::new("/tmp/a~b")), Path::new("/tmp/a~b"));
    }
}
