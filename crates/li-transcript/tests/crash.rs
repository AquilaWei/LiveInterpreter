//! Does the transcript actually survive being killed?
//!
//! Lines must be to be appended as they are finalised "so that an
//! unexpected shutdown keeps what was already recognised", and this is the
//! test of exactly that: content still there after `kill -9`. Every
//! plausible way of getting this wrong -- a `BufWriter`, a rewrite-on-close, a
//! record held back until its translation arrives -- passes the ordinary unit
//! tests and fails this one.
//!
//! The child is this same test binary, re-run with an environment variable that
//! makes [`the_child_writes_a_session`] do the writing instead of returning
//! immediately. `Child::kill` is SIGKILL on Unix and `TerminateProcess` on
//! Windows: no unwinding, no destructors, no `Drop for Writer`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use li_transcript::{Format, Langs, TranscriptConfig, TranscriptSink, Writer, jsonl};
use li_types::{Lane, TranscriptLine};

const DIR: &str = "LI_TRANSCRIPT_CRASH_DIR";
const LINES: u64 = 6;

fn line(line_id: u64) -> TranscriptLine {
    let start_s = line_id as f64 * 3.0;
    TranscriptLine {
        line_id,
        start_s,
        end_s: start_s + 2.0,
        source: format!("This is line number {line_id}."),
        translation: None,
        lane: Lane::Accurate,
        reason: None,
    }
}

#[test]
fn the_child_writes_a_session() {
    let Ok(dir) = std::env::var(DIR) else {
        return; // The parent run. Nothing to do.
    };
    let cfg = TranscriptConfig {
        enabled: true,
        dir: PathBuf::from(dir),
        formats: vec![Format::Txt, Format::Srt, Format::Vtt, Format::Jsonl],
        bilingual_file: true,
    };
    let mut w = Writer::open(&cfg, &Langs::default()).expect("open the transcript");
    for id in 1..=LINES {
        w.on_source_final(&line(id)).expect("write a line");
        w.on_translation_final(id, &format!("這是第 {id} 行。"))
            .expect("write a translation");
    }
    // Alive, holding the files open, with nothing flushed or closed. The parent
    // kills us here; `std::process::exit` is the fallback if it somehow does
    // not, so a broken test cannot leave a process spinning.
    std::thread::sleep(Duration::from_secs(30));
    std::process::exit(1);
}

#[test]
fn a_finalised_line_survives_kill_9() {
    if std::env::var(DIR).is_ok() {
        return; // We are the child; the test above is our job.
    }
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("crash");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["the_child_writes_a_session", "--exact", "--nocapture"])
        .env(DIR, &dir)
        .spawn()
        .expect("re-run this test binary as the child");

    let files = wait_for_files(&dir, 5, Duration::from_secs(30));
    child.kill().expect("SIGKILL the child");
    child.wait().unwrap();

    let by_ext = |ext: &str| -> String {
        let p = files
            .iter()
            .find(|p| p.extension().is_some_and(|e| e == ext) && !is_bilingual(p))
            .unwrap_or_else(|| panic!("no .{ext} among {files:?}"));
        std::fs::read_to_string(p).unwrap()
    };

    // The source transcript: every line, in order, nothing else.
    let txt = by_ext("txt");
    let want: String = (1..=LINES)
        .map(|i| format!("This is line number {i}.\n"))
        .collect();
    assert_eq!(txt, want, "the source transcript lost a line");

    // The subtitle files: complete blocks, real spans, no half-written cue.
    let srt = by_ext("srt");
    assert_eq!(srt.matches(" --> ").count(), LINES as usize, "{srt}");
    assert!(
        srt.ends_with(&format!("This is line number {LINES}.\n\n")),
        "{srt}"
    );
    assert!(
        !srt.contains("00:00:03,000 --> 00:00:03,000"),
        "zero-length cue: {srt}"
    );
    let vtt = by_ext("vtt");
    assert!(vtt.starts_with("WEBVTT\n\n"), "{vtt}");
    assert_eq!(vtt.matches(" --> ").count(), LINES as usize, "{vtt}");

    // The jsonl: uncollapsed, because `finish` never ran -- and every line and
    // every translation is still in it once folded.
    let path = files
        .iter()
        .find(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .unwrap();
    let raw = std::fs::read_to_string(path).unwrap();
    assert_eq!(
        raw.lines().count(),
        2 * LINES as usize,
        "expected the log form"
    );
    let folded = jsonl::read(path).expect("the log parses");
    assert_eq!(folded.len(), LINES as usize);
    for (i, rec) in folded.iter().enumerate() {
        let id = i as u64 + 1;
        assert_eq!(rec.line_id, id);
        assert_eq!(
            rec.source.as_deref(),
            Some(&*format!("This is line number {id}."))
        );
        assert_eq!(
            rec.translation.as_deref(),
            Some(&*format!("這是第 {id} 行。"))
        );
        assert_eq!(rec.lane, Some(Lane::Accurate));
    }

    // The bilingual file is the one place a crash does cost something: the last
    // pair is still queued, waiting to be written in order. It is the derived
    // file, and its source half is safe in the transcript above.
    let bi = files
        .iter()
        .find(|p| is_bilingual(p))
        .expect("a bilingual file");
    let bi = std::fs::read_to_string(bi).unwrap();
    assert!(
        bi.contains("This is line number 1.") && bi.contains("這是第 1 行。"),
        "{bi}"
    );
    assert!(
        bi.matches("This is line number").count() >= LINES as usize - 1,
        "the bilingual file should be at most one pair behind:\n{bi}"
    );
}

fn is_bilingual(p: &Path) -> bool {
    p.file_stem()
        .is_some_and(|s| s.to_string_lossy().ends_with("_en-zh"))
}

/// The child has to get far enough to have written something before it is any
/// use to kill it.
fn wait_for_files(dir: &Path, want: usize, timeout: Duration) -> Vec<PathBuf> {
    let deadline = Instant::now() + timeout;
    loop {
        let files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
            .unwrap_or_default();
        let written = files
            .iter()
            .filter(|p| std::fs::metadata(p).is_ok_and(|m| m.len() > 0))
            .count();
        if files.len() >= want && written >= want - 1 {
            return files;
        }
        assert!(
            Instant::now() < deadline,
            "the child never wrote: {files:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
