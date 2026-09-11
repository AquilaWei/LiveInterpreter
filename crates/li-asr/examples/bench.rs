//! Measure one lane against a wav file, the way task 1.0a/1.0b measured the
//! Python PoC, so the Rust numbers can be compared to them directly.
//!
//!     cargo run --release -p li-asr --example bench -- \
//!         --lane fast --wav testdata/ami_meeting.wav
//!     cargo run --release -p li-asr --example bench -- \
//!         --lane accurate --wav testdata/ami_meeting.wav --device vulkan
//!
//! The audio is streamed at playback speed and the source never slows down for
//! a slow consumer, so "wall time vs clip length" is a real answer to "does it
//! keep up". Latency is measured the way PLAN §7 defines it: from the moment
//! the sentence's audio has finished to the moment its text exists.
//!
//! The accurate lane needs a buffer discipline before it can be measured at
//! all, and that discipline is `li-stream`'s job (task 1.5). What runs here is
//! the minimum that makes the engine measurable -- VAD-driven flush at a pause,
//! a hard cap on the buffer -- and it reports the mean buffer length, because
//! that is the quantity task 1.0b found to be the difference between working
//! and collapsing.

use std::{
    collections::VecDeque,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use li_asr::{DeviceRequest, LaneSpec};
use li_audio::file::FileSource;
use li_types::AsrEvent;
use li_vad::{GateConfig, SileroVad, Vad, VadEvent};

const SAMPLE_RATE: f64 = 16_000.0;
/// PLAN §12 defaults. Duplicated rather than imported: `li-stream` owns the
/// policy, and this file is a measuring stick, not a second implementation.
const TICK: Duration = Duration::from_millis(800);
const PAUSE_FLUSH: Duration = Duration::from_millis(600);
const MAX_BUFFER_S: f64 = 12.0;

struct Args {
    lane: String,
    wav: PathBuf,
    model: Option<PathBuf>,
    device: DeviceRequest,
    threads: Option<usize>,
    /// Accurate lane only. `None` = the lane's own policy (crop on the CPU,
    /// full window on a GPU); `Some(b)` forces it, for the accuracy comparison.
    crop_ctx: Option<bool>,
    /// Accurate lane only: turn whisper's temperature fallback -- its
    /// repetition guard -- off, to keep the cost of that measurable.
    no_fallback: bool,
    out: Option<PathBuf>,
}

fn parse() -> Result<Args> {
    let mut a = Args {
        lane: "fast".into(),
        wav: PathBuf::new(),
        model: None,
        device: DeviceRequest::Auto,
        threads: None,
        crop_ctx: None,
        no_fallback: false,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("a flag is missing its value"))
        };
        match flag.as_str() {
            "--lane" => a.lane = val()?,
            "--wav" => a.wav = val()?.into(),
            "--model" => a.model = Some(val()?.into()),
            "--device" => a.device = val()?.parse()?,
            "--threads" => a.threads = Some(val()?.parse()?),
            "--full-ctx" => a.crop_ctx = Some(false),
            "--crop-ctx" => a.crop_ctx = Some(true),
            "--no-fallback" => a.no_fallback = true,
            "--out" => a.out = Some(val()?.into()),
            other => bail!("unknown flag {other}"),
        }
    }
    if a.wav.as_os_str().is_empty() {
        bail!(
            "usage: --wav <clip.wav> [--lane fast|accurate] [--model P] \
             [--device D] [--threads N] [--full-ctx|--crop-ctx] [--no-fallback] \
             [--out F]"
        );
    }
    Ok(a)
}

fn default_model(lane: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    let cache = PathBuf::from(home).join(".cache/liveinterpreter/models");
    match lane {
        "fast" => cache.join("sherpa-onnx-streaming-zipformer-en-2023-06-21"),
        _ => cache.join("ggml/ggml-small.en-q5_1.bin"),
    }
}

/// Peak resident set, straight from the kernel. Sampling a running process
/// would miss the peak between samples; `VmHWM` cannot.
fn peak_rss_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[i]
}

fn row(label: &str, values: &[f64]) -> String {
    if values.is_empty() {
        return format!("| {label} | – | – | – |");
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    format!(
        "| {label} | {:.2} | {:.2} | {:.2} |",
        pct(&v, 0.5),
        pct(&v, 0.9),
        pct(&v, 1.0)
    )
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let args = parse()?;
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| default_model(&args.lane));

    // The fast lane is budgeted at about one core (§8); the accurate lane gets
    // what is left, because it is the one that has to keep up with the clock.
    let threads = args.threads.unwrap_or(match args.lane.as_str() {
        "fast" => 2,
        _ => std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(2).max(1))
            .unwrap_or(4),
    });

    let mut spec = LaneSpec::new(
        match args.lane.as_str() {
            "fast" => "sherpa",
            "accurate" => "whispercpp",
            other => bail!("--lane wants `fast` or `accurate`, not {other}"),
        },
        &model,
    );
    spec.device = args.device;
    spec.threads = threads;

    let load = Instant::now();
    let mut engine: Box<dyn li_asr::AsrEngine> = if args.crop_ctx.is_some() || args.no_fallback {
        let mut w = li_asr::whisper::WhisperAccurate::open(&spec)?;
        if let Some(crop) = args.crop_ctx {
            w = w.with_cropped_context(crop);
        }
        if args.no_fallback {
            w = w.with_temperature_fallback(false);
        }
        Box::new(w)
    } else {
        li_asr::build(&spec)?
    };
    let load_s = load.elapsed().as_secs_f64();
    let backend = engine.backend().to_string();

    let mut rx = FileSource::new(&args.wav).with_agc(true).open()?;
    let t0 = Instant::now();

    let mut audio_samples: u64 = 0;
    let mut lines: Vec<(String, f64)> = Vec::new(); // text, latency
    let mut finals = 0usize;
    let mut passes = 0usize;
    let mut engine_ms: Vec<f64> = Vec::new();

    // Accurate-lane driver state; the fast lane segments itself.
    let mut buffer: VecDeque<f32> = VecDeque::new();
    let mut buf_origin = Duration::ZERO;
    let mut last_tick = Instant::now();
    let mut vad = if args.lane == "accurate" {
        Some(SileroVad::new(GateConfig::default())?)
    } else {
        None
    };
    let mut buf_lengths: Vec<f64> = Vec::new();
    let mut overruns = 0usize;

    while let Some(frame) = rx.recv().await {
        let t_origin = Duration::from_secs_f64(audio_samples as f64 / SAMPLE_RATE);
        audio_samples += frame.pcm.len() as u64;
        let t_now = Duration::from_secs_f64(audio_samples as f64 / SAMPLE_RATE);

        if let Some(vad) = vad.as_mut() {
            let events = vad.push(&frame)?;
            buffer.extend(frame.pcm.iter().copied());
            let buf_s = buffer.len() as f64 / SAMPLE_RATE;
            let ended = events.iter().any(|e| matches!(e, VadEvent::SpeechEnd));
            let flush = (ended && vad.silence() >= PAUSE_FLUSH) || buf_s >= MAX_BUFFER_S;

            if last_tick.elapsed() >= TICK || flush {
                buf_lengths.push(buf_s);
                let pcm: Vec<f32> = buffer.iter().copied().collect();
                let started = Instant::now();
                let evs = engine.feed(&pcm, buf_origin).await?;
                let took = started.elapsed();
                engine_ms.push(took.as_secs_f64() * 1e3);
                passes += evs.len();
                if took >= TICK {
                    overruns += 1;
                }
                // Restart the clock *after* the pass, not before it. Starting it
                // before means a pass that overruns the tick re-triggers on the
                // very next frame and the lane runs flat out -- the positive
                // feedback loop task 1.0b diagnosed in the Phase 0 PoC, faithfully
                // reproduced by this file on its first run. Backing off properly
                // (shrinking the buffer, stretching the tick) is `li-stream`'s
                // job in task 1.5; this only stops the measuring stick bending.
                last_tick = Instant::now();
            }
            if flush {
                if let Some(AsrEvent::Final { words, t_end, .. }) = engine.finalize().await? {
                    finals += 1;
                    let latency = t0.elapsed().as_secs_f64() - t_end.as_secs_f64();
                    lines.push((li_asr::words::text_of(&words), latency));
                }
                buffer.clear();
                buf_origin = t_now;
            }
        } else {
            let started = Instant::now();
            let evs = engine.feed(&frame.pcm, t_origin).await?;
            engine_ms.push(started.elapsed().as_secs_f64() * 1e3);
            for ev in evs {
                match ev {
                    AsrEvent::Partial { .. } => passes += 1,
                    AsrEvent::Final { words, t_end, .. } => {
                        finals += 1;
                        let latency = t0.elapsed().as_secs_f64() - t_end.as_secs_f64();
                        lines.push((li_asr::words::text_of(&words), latency));
                    }
                }
            }
        }
    }
    if let Some(AsrEvent::Final { words, t_end, .. }) = engine.finalize().await? {
        finals += 1;
        lines.push((
            li_asr::words::text_of(&words),
            t0.elapsed().as_secs_f64() - t_end.as_secs_f64(),
        ));
    }

    let wall = t0.elapsed().as_secs_f64();
    let audio_s = audio_samples as f64 / SAMPLE_RATE;
    let latencies: Vec<f64> = lines.iter().map(|(_, l)| *l).collect();
    let work = engine_ms.iter().sum::<f64>() / 1e3 / audio_s;

    let mut out = String::new();
    let name = args.wav.file_name().unwrap_or_default().to_string_lossy();
    out.push_str(&format!("# li-asr bench — {name}\n\n"));
    out.push_str(&format!("- lane: **{}** — {backend}\n", args.lane));
    out.push_str(&format!(
        "- model load: {load_s:.1}s | {threads} threads{}{}\n",
        match args.crop_ctx {
            Some(true) => " | forced cropped encoder context",
            Some(false) => " | forced full 30s encoder context",
            None => "",
        },
        if args.no_fallback {
            " | temperature fallback OFF"
        } else {
            ""
        }
    ));
    out.push_str(&format!(
        "- audio {audio_s:.1}s, wall {wall:.1}s ({:+.1}%) | sentences {finals} | passes {passes}\n",
        (wall / audio_s - 1.0) * 100.0
    ));
    if let Some(rss) = peak_rss_mib() {
        out.push_str(&format!("- **peak RSS: {rss:.0} MiB**\n"));
    }
    if !buf_lengths.is_empty() {
        let mean = buf_lengths.iter().sum::<f64>() / buf_lengths.len() as f64;
        let max = buf_lengths.iter().copied().fold(0.0, f64::max);
        out.push_str(&format!(
            "- rolling buffer: **mean {mean:.1}s** (PLAN §12 requires <= 6s), max {max:.1}s\n"
        ));
    }
    out.push_str(&format!(
        "- engine work per audio second: **{work:.3}** (1.0 = exactly keeping up)\n"
    ));
    if !buf_lengths.is_empty() {
        out.push_str(&format!(
            "- passes that overran the {:.1}s tick: {overruns} of {}\n",
            TICK.as_secs_f64(),
            engine_ms.len()
        ));
    }
    out.push('\n');
    out.push_str("| metric | median | p90 | max |\n|---|---|---|---|\n");
    out.push_str(&row("sentence latency (s)", &latencies));
    out.push('\n');
    out.push_str(&row("one engine call (ms)", &engine_ms));
    out.push('\n');

    if args.lane == "fast" {
        let mut s = latencies.clone();
        s.sort_by(f64::total_cmp);
        let verdict = if !s.is_empty() && pct(&s, 0.5) <= 1.2 && pct(&s, 0.9) <= 2.0 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        out.push_str(&format!(
            "\nGate G1 (median <=1.2s, p90 <=2.0s): **{verdict}**\n"
        ));
    }
    out.push_str("\n## Transcript\n\n");
    out.push_str(
        &lines
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    );
    out.push('\n');

    println!("{out}");
    if let Some(p) = &args.out {
        std::fs::write(p, &out)?;
        println!("[written to {}]", p.display());
    }
    Ok(())
}
