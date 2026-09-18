//! `cargo xtask pause` -- is a pause shorter than `rule2`'s 0.6 s a place a
//! line could have been cut?
//!
//! The complaint this answers is "他已經斷句但我們好像抓不到": a speaker going
//! fast still ends sentences, but ends them with a 0.3 s breath rather than a
//! 0.6 s one. sherpa's `rule2_min_trailing_silence` never fires, so the line
//! runs on until either a real pause arrives or `rule3` gives up at 20 s. On
//! the user's own 2026-09-04 sessions that cost a median of 3.95 s of extra
//! waiting, and six lines hit the 20 s cap outright.
//!
//! ## Why the VAD and not the word times
//!
//! The obvious measurement -- the gap between two words -- does not exist.
//! sherpa timestamps token *onsets*, and [`li_asr::words::merge`] fills each
//! word's `end` with the next word's `start` (`words.rs:63-65`), so every
//! inter-word gap in a `Vec<Word>` is exactly zero by construction. The silence
//! itself is only visible to something that looks at the audio, which is the
//! VAD -- already running on every frame in `li_core::Engine` and already
//! reporting the length of the current trailing-silence run.
//!
//! So this walks the clip with the fast lane and the VAD side by side and
//! writes down every run of silence: how long it lasted, how many words were
//! sitting unclosed on screen when it began, and -- the number that matters --
//! how much longer the line actually took to close after it.

use std::{path::PathBuf, time::Duration};

use anyhow::{Result, bail};
use li_asr::{DeviceRequest, LaneSpec};
use li_audio::file::FileSource;
use li_types::AsrEvent;
use li_vad::{GateConfig, SileroVad, Vad};

const SAMPLE_RATE: f64 = 16_000.0;

/// Only runs at least this long are written down. Below it there is nothing to
/// discuss: the VAD's own `min_silence` is 200 ms, so shorter runs are not
/// reported as speech ending at all.
const FLOOR: Duration = Duration::from_millis(200);

pub struct Args {
    wav: PathBuf,
    fast_model: Option<PathBuf>,
    /// Candidate thresholds, in seconds.
    thresholds: Vec<f64>,
    /// Don't cut a line this short even at a long pause.
    min_words: usize,
    dump: bool,
    endpoint_silence: Option<f32>,
    max_utterance: Option<f32>,
    /// Print each closed segment's inter-word onset deltas instead of the VAD
    /// report: the second question, asked of sherpa's own clock.
    segments: bool,
}

pub fn parse(mut it: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args {
        wav: PathBuf::new(),
        fast_model: None,
        thresholds: vec![0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        min_words: 4,
        dump: false,
        endpoint_silence: None,
        max_utterance: None,
        segments: false,
    };
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("a flag is missing its value"))
        };
        match flag.as_str() {
            "--wav" => a.wav = val()?.into(),
            "--fast-model" => a.fast_model = Some(val()?.into()),
            "--min-words" => a.min_words = val()?.parse()?,
            "--thresholds" => {
                a.thresholds = val()?
                    .split(',')
                    .map(|s| s.trim().parse::<f64>())
                    .collect::<Result<_, _>>()?
            }
            "--dump" => a.dump = true,
            "--endpoint-silence" => a.endpoint_silence = Some(val()?.parse()?),
            "--max-utterance" => a.max_utterance = Some(val()?.parse()?),
            "--segments" => a.segments = true,
            other => bail!("unknown flag {other}"),
        }
    }
    if a.wav.as_os_str().is_empty() {
        bail!(
            "usage: cargo xtask pause --wav <clip.wav> [--fast-model P] \
             [--thresholds 0.25,0.3,...] [--min-words N] [--dump] [--segments]"
        );
    }
    Ok(a)
}

/// The shared model cache, by the one rule that decides it.
///
/// Spelled out by hand here until the Flatpak work: three copies of
/// `$HOME/.cache/liveinterpreter/models`, which is the exact duplication
/// `li_types::paths` was written to remove (see its module docs). They went
/// wrong together when the flatpak work taught that rule about
/// `XDG_CACHE_HOME`, and a harness that looks somewhere the app does not is a
/// harness that measures nothing.
fn cache() -> PathBuf {
    li_types::paths::model_cache()
}

/// One run of trailing silence, as the VAD saw it.
struct Run {
    /// Where the silence began on the audio timeline.
    start: Duration,
    len: Duration,
    /// Words the fast lane had decoded but not yet closed when it began.
    words_open: usize,
    /// The unclosed text, for reading with `--dump`.
    open: String,
    /// The run reached `rule2` and sherpa closed the line by itself.
    closed_here: bool,
    /// When the line the run interrupted actually did close.
    closed_at: Option<Duration>,
}

pub fn run(args: Args) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(go(args))
}

async fn go(args: Args) -> Result<()> {
    let mut spec = LaneSpec::new(
        "sherpa",
        args.fast_model
            .clone()
            .unwrap_or_else(|| cache().join("sherpa-onnx-streaming-zipformer-en-2023-06-21")),
    );
    spec.device = DeviceRequest::Cpu;
    spec.threads = 2;
    if let Some(v) = args.endpoint_silence {
        spec.endpoint_silence_s = v;
    }
    if let Some(v) = args.max_utterance {
        spec.max_utterance_s = v;
    }
    let mut fast = li_asr::build(&spec)?;
    let mut vad = SileroVad::new(GateConfig::default())?;

    // Not realtime: nothing here is a latency measurement, so there is no
    // reason to sit through the clip.
    let mut rx = FileSource::new(&args.wav)
        .with_agc(true)
        .with_realtime(false)
        .open()?;

    let mut samples: u64 = 0;
    let mut runs: Vec<Run> = Vec::new();
    let mut open = String::new();
    let mut in_run = false;
    // Runs still waiting to learn when their line closed.
    let mut unclosed: Vec<usize> = Vec::new();
    let mut endpoints = 0usize;
    let mut segments: Vec<Vec<li_types::Word>> = Vec::new();

    while let Some(frame) = rx.recv().await {
        let t_origin = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
        samples += frame.pcm.len() as u64;
        let now = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);

        vad.push(&frame)?;
        let silence = vad.silence();

        for ev in fast.feed(&frame.pcm, t_origin).await? {
            match ev {
                AsrEvent::Partial { text, .. } => open = text,
                AsrEvent::Final { ref words, .. } => {
                    endpoints += 1;
                    segments.push(words.clone());
                    open.clear();
                    // Every run that was still waiting now knows its answer.
                    for i in unclosed.drain(..) {
                        runs[i].closed_at = Some(now);
                    }
                    if let Some(last) = runs.last_mut()
                        && !last.closed_here
                        && last.start + last.len + Duration::from_millis(120) >= now
                    {
                        // The endpoint landed inside this very run: it is
                        // `rule2` firing, not a later sentence.
                        last.closed_here = true;
                        last.closed_at = Some(now);
                    }
                }
            }
        }

        if silence >= FLOOR {
            if !in_run {
                in_run = true;
                runs.push(Run {
                    start: now.saturating_sub(silence),
                    len: silence,
                    words_open: open.split_whitespace().count(),
                    open: open.clone(),
                    closed_here: false,
                    closed_at: None,
                });
                unclosed.push(runs.len() - 1);
            } else if let Some(last) = runs.last_mut() {
                last.len = silence;
            }
        } else {
            in_run = false;
        }
    }

    if let Some(AsrEvent::Final { words, .. }) = fast.finalize().await? {
        endpoints += 1;
        segments.push(words);
        let end = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
        for i in unclosed.drain(..) {
            runs[i].closed_at = Some(end);
        }
    }

    if args.segments {
        report_segments(&segments);
    } else {
        report(&args, &runs, endpoints);
    }
    Ok(())
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[i]
}

fn stats(mut v: Vec<f64>) -> (usize, f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v.len(), pct(&v, 0.5), pct(&v, 0.9), pct(&v, 1.0))
}

fn report(args: &Args, runs: &[Run], endpoints: usize) {
    println!("clip {}", args.wav.display());
    println!(
        "silence runs >= {:.0} ms: {}",
        FLOOR.as_secs_f64() * 1e3,
        runs.len()
    );
    println!("endpoints sherpa found: {endpoints}\n");

    let (real, short): (Vec<&Run>, Vec<&Run>) = runs.iter().partition(|r| r.closed_here);
    let (n, p50, p90, max) = stats(real.iter().map(|r| r.len.as_secs_f64()).collect());
    println!(
        "runs that reached rule2 and closed a line: n={n} p50={p50:.2}s p90={p90:.2}s max={max:.2}s"
    );
    let (n, p50, p90, max) = stats(short.iter().map(|r| r.len.as_secs_f64()).collect());
    println!(
        "runs that did not:                          n={n} p50={p50:.2}s p90={p90:.2}s max={max:.2}s\n"
    );

    println!("| cut at | extra cuts | words at cut p50 | wait saved p50 | p90 | max |");
    println!("|---|---|---|---|---|---|");
    for &t in &args.thresholds {
        let t = Duration::from_secs_f64(t);
        let hits: Vec<&&Run> = short
            .iter()
            .filter(|r| r.len >= t && r.words_open >= args.min_words)
            .collect();
        let saved: Vec<f64> = hits
            .iter()
            .filter_map(|r| {
                let closed = r.closed_at?;
                Some(closed.saturating_sub(r.start + t).as_secs_f64())
            })
            .collect();
        let words = stats(hits.iter().map(|r| r.words_open as f64).collect());
        let (_, p50, p90, max) = stats(saved.clone());
        println!(
            "| {:.2}s | {} | {:.0} | {:.2}s | {:.2}s | {:.2}s |",
            t.as_secs_f64(),
            hits.len(),
            words.1,
            p50,
            p90,
            max
        );
    }

    if args.dump {
        println!("\nrun start | len | words | closed after | text");
        for r in runs {
            println!(
                "{:8.2} | {:.2} | {:3} | {:>6} | {}",
                r.start.as_secs_f64(),
                r.len.as_secs_f64(),
                r.words_open,
                r.closed_at.map_or("-".into(), |c| format!(
                    "{:.2}",
                    c.saturating_sub(r.start).as_secs_f64()
                )),
                if r.closed_here { "[rule2]" } else { &r.open }
            );
        }
    }
}

/// The other clock: how far apart are two words' onsets inside one segment?
///
/// This is what the VAD cannot see. sherpa timestamps every token it emits, so
/// a stretch where it emitted nothing shows up as a wide interval between two
/// onsets -- including stretches far too short to be `rule2`'s 0.6 s, which are
/// exactly the ones a fast speaker's sentence breaks land in. The interval
/// carries the previous word's own duration as well, so it is an upper bound on
/// the silence rather than the silence itself; a threshold read off it has to
/// be read off *this* number, not off a pause length.
fn report_segments(segments: &[Vec<li_types::Word>]) {
    let mut all: Vec<f64> = Vec::new();
    println!("| segment | span | words | widest onset gaps (s @ word) |");
    println!("|---|---|---|---|");
    for (i, ws) in segments.iter().enumerate() {
        if ws.len() < 2 {
            continue;
        }
        let mut gaps: Vec<(f64, usize)> = (1..ws.len())
            .map(|j| {
                (
                    (ws[j].start.as_secs_f64() - ws[j - 1].start.as_secs_f64()),
                    j,
                )
            })
            .collect();
        all.extend(gaps.iter().map(|(g, _)| *g));
        gaps.sort_by(|a, b| b.0.total_cmp(&a.0));
        let span = ws.last().unwrap().end.as_secs_f64() - ws[0].start.as_secs_f64();
        let top: Vec<String> = gaps
            .iter()
            .take(3)
            .map(|(g, j)| format!("{g:.2}@{}", ws[*j].text))
            .collect();
        println!(
            "| {i} | {:.1}-{:.1} ({span:.1}s) | {} | {} |",
            ws[0].start.as_secs_f64(),
            ws.last().unwrap().end.as_secs_f64(),
            ws.len(),
            top.join("  ")
        );
    }
    let (n, p50, p90, max) = stats(all.clone());
    let mut v = all;
    v.sort_by(f64::total_cmp);
    println!(
        "\nonset gaps inside a segment: n={n} p50={p50:.2}s p90={p90:.2}s p99={:.2}s max={max:.2}s",
        pct(&v, 0.99)
    );
    for t in [0.3, 0.4, 0.5, 0.6, 0.8] {
        println!(
            "  >= {t:.1}s: {} ({:.1}%)",
            v.iter().filter(|g| **g >= t).count(),
            100.0 * v.iter().filter(|g| **g >= t).count() as f64 / v.len().max(1) as f64
        );
    }
}
