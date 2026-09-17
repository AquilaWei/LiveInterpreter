//! `cargo xtask eval` -- the acceptance harness of PLAN §16.
//!
//! Runs the whole recognition pipeline over a golden clip at playback speed and
//! reports the gate numbers. The clip is never slowed down for a slow consumer,
//! so "wall time vs clip length" is a real answer to "does this keep up", and a
//! lane that falls behind loses audio exactly as it would in a meeting.
//!
//! This is the first measurement that includes `li-stream`, and therefore the
//! first that can speak to G2a/G2b at all: task 1.4's bench finalised on every
//! VAD pause, which is not the commit policy the product ships.
//!
//! Task 1.15 extends this to the translation gates; the ASR half is here
//! because task 1.5 claims G2b and a claimed gate has to be measured.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use li_asr::{DeviceRequest, LaneSpec, window::Ring};
use li_audio::file::FileSource;
use li_stream::{
    LaneMode, Stream, StreamConfig,
    load::Load,
    punct::{BoundaryConfig, BoundaryPolicy},
};
use li_transcript::{Langs, TranscriptConfig, TranscriptSink, Writer};
use li_types::{AsrEvent, EngineEvent, FastReason, Lane, TranscriptLine};
use li_vad::{GateConfig, SileroVad, Vad, VadEvent};

const SAMPLE_RATE: f64 = 16_000.0;

pub struct Args {
    wav: PathBuf,
    reference: Option<PathBuf>,
    mode: LaneMode,
    fast_model: Option<PathBuf>,
    accurate_model: Option<PathBuf>,
    device: DeviceRequest,
    out: Option<PathBuf>,
    transcript: Option<PathBuf>,
    /// Overrides for the two segmentation knobs task 1.24 sweeps.
    endpoint_silence: Option<f32>,
    max_utterance: Option<f32>,
    /// Restore punctuation and casing on the fast lane, as `li_core::Engine`
    /// does when `[asr.fast] punctuation` is on. Off here, because this is the
    /// harness that has to be able to show the two runs are the same.
    punct: bool,
    /// End a line where the restored punctuation says a sentence ended, as
    /// `[asr.fast] semantic_cut` does (task 1.25 stage 2). Implies `--punct`:
    /// with no marks there is nothing to cut on.
    cut: bool,
}

pub fn parse(mut it: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args {
        wav: PathBuf::new(),
        reference: None,
        mode: LaneMode::Dual,
        fast_model: None,
        accurate_model: None,
        device: DeviceRequest::Auto,
        out: None,
        transcript: None,
        endpoint_silence: None,
        max_utterance: None,
        punct: false,
        cut: false,
    };
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("a flag is missing its value"))
        };
        match flag.as_str() {
            "--wav" => a.wav = val()?.into(),
            "--ref" => a.reference = Some(val()?.into()),
            "--lanes" => {
                a.mode = match val()?.as_str() {
                    "dual" => LaneMode::Dual,
                    "fast" => LaneMode::FastOnly,
                    "accurate" => LaneMode::AccurateOnly,
                    other => bail!("--lanes wants dual|fast|accurate, not {other}"),
                }
            }
            "--fast-model" => a.fast_model = Some(val()?.into()),
            "--accurate-model" => a.accurate_model = Some(val()?.into()),
            "--device" => a.device = val()?.parse()?,
            "--out" => a.out = Some(val()?.into()),
            "--transcript" => a.transcript = Some(val()?.into()),
            "--endpoint-silence" => a.endpoint_silence = Some(val()?.parse()?),
            "--max-utterance" => a.max_utterance = Some(val()?.parse()?),
            "--punct" => a.punct = true,
            "--semantic-cut" => {
                a.cut = true;
                a.punct = true;
            }
            other => bail!("unknown flag {other}"),
        }
    }
    if a.wav.as_os_str().is_empty() {
        bail!(
            "usage: cargo xtask eval --wav <clip.wav> [--ref <clip.en.txt>] \
             [--lanes dual|fast|accurate] [--fast-model P] [--accurate-model P] \
             [--device D] [--out F] [--transcript DIR] \
             [--endpoint-silence S] [--max-utterance S] [--punct] [--semantic-cut]"
        );
    }
    Ok(a)
}

/// The shared model cache, by the one rule that decides it.
///
/// Spelled out by hand here until task 1.14b: three copies of
/// `$HOME/.cache/liveinterpreter/models`, which is the exact duplication
/// `li_types::paths` was written to remove (see its module docs). They went
/// wrong together when the flatpak work taught that rule about
/// `XDG_CACHE_HOME`, and a harness that looks somewhere the app does not is a
/// harness that measures nothing.
fn cache() -> PathBuf {
    li_types::paths::model_cache()
}

/// `testdata/foo.wav` -> `testdata/foo.en.txt`, the convention the golden set
/// already uses.
fn reference_for(wav: &Path) -> PathBuf {
    let stem = wav.file_stem().unwrap_or_default().to_string_lossy();
    wav.with_file_name(format!("{stem}.en.txt"))
}

fn peak_rss_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: f64 = s
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
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
        return format!("| {label} | – | – | – | – |\n");
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    format!(
        "| {label} | {} | {:.2} | {:.2} | {:.2} |\n",
        v.len(),
        pct(&v, 0.5),
        pct(&v, 0.9),
        pct(&v, 1.0)
    )
}

/// Transcribe exactly one utterance, cut out of the raw audio history.
///
/// A little padding either side: the two engines place a boundary slightly
/// differently, and a clipped first phoneme is a mis-heard word.
async fn segment(
    acc: &mut dyn li_asr::AsrEngine,
    ring: &Ring,
    t_start: Duration,
    t_end: Duration,
    prompt: &str,
    debug: bool,
) -> Result<(Vec<li_types::Word>, Duration)> {
    // The same window `li_core::Engine` hands it, by construction: the harness
    // that measures the accurate lane has to feed it what the product will.
    let (pcm, a) = ring.cut(t_start, t_end);
    let b = a + Duration::from_secs_f64(pcm.len() as f64 / SAMPLE_RATE);

    acc.set_prompt(prompt);
    let started = Instant::now();
    let evs = acc.feed(&pcm, a).await?;
    let took = started.elapsed();
    let hyp = evs
        .into_iter()
        .find_map(|e| match e {
            AsrEvent::Partial { words, .. } => Some(words),
            _ => None,
        })
        .unwrap_or_default();
    if debug {
        eprintln!(
            "[segment] {:.1}..{:.1} ({:.1}s, {:.0}ms) | {}",
            a.as_secs_f64(),
            b.as_secs_f64(),
            pcm.len() as f64 / SAMPLE_RATE,
            took.as_secs_f64() * 1e3,
            hyp.iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    Ok((hyp, took))
}

#[derive(Debug)]
struct Line {
    text: String,
    lane: Lane,
    reason: Option<FastReason>,
    latency: f64,
}

pub fn run(args: Args) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(go(args))
}

async fn go(args: Args) -> Result<()> {
    let debug = std::env::var("LI_EVAL_DEBUG").is_ok();
    let cfg = StreamConfig::default();
    let pause_flush = Duration::from_secs_f64(cfg.pause_flush_s);
    let max_segment = Duration::from_secs_f64(cfg.max_segment_s);

    let mut fast = match args.mode {
        LaneMode::AccurateOnly => None,
        _ => {
            let mut spec = LaneSpec::new(
                "sherpa",
                args.fast_model.clone().unwrap_or_else(|| {
                    cache().join("sherpa-onnx-streaming-zipformer-en-2023-06-21")
                }),
            );
            spec.device = DeviceRequest::Cpu;
            spec.threads = 2;
            if let Some(s) = args.endpoint_silence {
                spec.endpoint_silence_s = s;
            }
            if let Some(s) = args.max_utterance {
                spec.max_utterance_s = s;
            }
            Some(li_asr::build(&spec)?)
        }
    };
    let mut accurate = match args.mode {
        LaneMode::FastOnly => None,
        _ => {
            let mut spec = LaneSpec::new(
                "whispercpp",
                args.accurate_model
                    .clone()
                    .unwrap_or_else(|| cache().join("ggml/ggml-small.en-q5_1.bin")),
            );
            spec.device = args.device;
            spec.threads = std::thread::available_parallelism()
                .map(|n| n.get().saturating_sub(2).max(1))
                .unwrap_or(4);
            Some(li_asr::build(&spec)?)
        }
    };
    let backends: Vec<String> = [fast.as_ref(), accurate.as_ref()]
        .into_iter()
        .flatten()
        .map(|e| e.backend().to_string())
        .collect();

    // The same restorer `li_core::Engine` builds, opened the same way, so a
    // `--punct` run is the shipped path and not an imitation of it.
    let mut punct = if args.punct {
        Some(li_asr::punct::OnlinePunct::open(
            &li_asr::punct::default_model_dir(),
            2,
        )?)
    } else {
        None
    };

    let mut stream = Stream::new(cfg, args.mode);
    // The same policy `li_core::Engine` builds, with the same thresholds.
    let mut cut = args
        .cut
        .then(|| BoundaryPolicy::new(BoundaryConfig::default()));
    let mut cuts = 0usize;
    let mut cut_declined = 0usize;
    // Audio time of each early cut in the segment still open. The gain is not
    // knowable when the cut is made -- it is how much later the endpoint
    // detector would have closed the line, and that has not happened yet -- so
    // they wait here until it does.
    let mut open_cuts: Vec<Duration> = Vec::new();
    let mut early_s: Vec<f64> = Vec::new();
    let mut load = Load::new();
    let mut vad = SileroVad::new(GateConfig::default())?;

    let mut rx = FileSource::new(&args.wav).with_agc(true).open()?;
    let t0 = Instant::now();

    let mut samples: u64 = 0;
    let mut lines: Vec<Line> = Vec::new();
    let mut fast_screen: Vec<f64> = Vec::new();
    let mut pass_ms: Vec<f64> = Vec::new();
    let mut forced_flushes = 0usize;
    let mut ring = Ring::default();
    let mut speech_start: Option<Duration> = None;

    // `--transcript` runs the real `li-transcript` writer over a real clip.
    // Until task 1.8 assembles the engine this is the only place the two meet,
    // and a golden clip is a better test of the subtitle timings than anything
    // hand-written: the spans come from the fast lane's own word times.
    let mut sink = args
        .transcript
        .as_ref()
        .map(|dir| -> Result<Writer> {
            let cfg = TranscriptConfig {
                enabled: true,
                dir: dir.clone(),
                formats: vec![
                    li_transcript::Format::Txt,
                    li_transcript::Format::Srt,
                    li_transcript::Format::Vtt,
                    li_transcript::Format::Jsonl,
                ],
                bilingual_file: false,
            };
            Writer::open(&cfg, &Langs::default())
        })
        .transpose()?;

    let collect = |events: Vec<EngineEvent>,
                   lines: &mut Vec<Line>,
                   sink: &mut Option<Writer>|
     -> Result<()> {
        for ev in events {
            if let EngineEvent::SourceFinal {
                line_id,
                text,
                t_start,
                t_end,
                lane,
                reason,
            } = ev
            {
                if let Some(w) = sink.as_mut() {
                    w.on_source_final(&TranscriptLine {
                        line_id,
                        start_s: t_start.as_secs_f64(),
                        end_s: t_end.as_secs_f64(),
                        source: text.clone(),
                        translation: None,
                        lane,
                        reason,
                    })?;
                }
                lines.push(Line {
                    text,
                    lane,
                    reason,
                    latency: t0.elapsed().as_secs_f64() - t_end.as_secs_f64(),
                });
            }
        }
        Ok(())
    };

    while let Some(frame) = rx.recv().await {
        let t_origin = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
        samples += frame.pcm.len() as u64;
        let now = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);

        let vad_events = vad.push(&frame)?;
        let ended = vad_events.iter().any(|e| matches!(e, VadEvent::SpeechEnd));
        let pause = ended && vad.silence() >= pause_flush;

        if let Some(fast) = fast.as_mut() {
            ring.push(&frame.pcm);
            for ev in fast.feed(&frame.pcm, t_origin).await? {
                match ev {
                    AsrEvent::Partial { text, .. } => {
                        let text = match punct.as_mut() {
                            Some(p) => p.restore(&text)?,
                            None => text,
                        };
                        if let Some(policy) = cut.as_mut()
                            && let Some(i) = policy.observe(&text, now)
                        {
                            let words = fast.open_words();
                            if words.len() != text.split_whitespace().count() {
                                cut_declined += 1;
                            } else if let (Some(next), Some(prefix)) =
                                (words.get(i + 1), words.get(policy.emitted()..=i))
                            {
                                let (prefix, t_end) = (prefix.to_vec(), next.start);
                                policy.cut(i, now);
                                cuts += 1;
                                open_cuts.push(now);
                                collect(
                                    stream.fast_cut(&prefix, t_end, now),
                                    &mut lines,
                                    &mut sink,
                                )?;
                            }
                        }
                        let shown = match cut.as_ref().map(BoundaryPolicy::emitted) {
                            Some(n) if n > 0 => text
                                .split_whitespace()
                                .skip(n)
                                .collect::<Vec<_>>()
                                .join(" "),
                            _ => text,
                        };
                        collect(stream.fast_partial(&shown, now), &mut lines, &mut sink)?;
                    }
                    AsrEvent::Final {
                        words,
                        t_start,
                        t_end,
                        ..
                    } => {
                        fast_screen.push(t0.elapsed().as_secs_f64() - t_end.as_secs_f64());
                        let words = match punct.as_mut() {
                            Some(p) => p.restore_words(&words)?,
                            None => words,
                        };
                        if let Some(policy) = cut.as_mut() {
                            policy.settle(now);
                            early_s.extend(open_cuts.drain(..).map(|c| (now - c).as_secs_f64()));
                        }
                        collect(stream.fast_final(&words, now), &mut lines, &mut sink)?;

                        // The accurate lane runs once over exactly this
                        // utterance. Its boundaries come from the fast lane's
                        // own endpoint detector, so the window is a whole
                        // clause -- which is the condition whisper is good at
                        // and the condition task 1.5 measured a re-transcribing
                        // buffer failing to provide.
                        if let Some(acc) = accurate.as_mut() {
                            let prompt = stream.prompt_tail();
                            let (hyp, took) = segment(
                                acc.as_mut(),
                                &ring,
                                t_start,
                                t_end.min(now),
                                &prompt,
                                debug,
                            )
                            .await?;
                            load.observe(t_end.saturating_sub(t_start), took);
                            pass_ms.push(took.as_secs_f64() * 1e3);
                            collect(stream.accurate_segment(&hyp, now), &mut lines, &mut sink)?;
                        }
                    }
                }
            }
        }

        // Accurate-only mode has no fast lane to borrow endpoints from, so the
        // VAD marks the utterances instead. Same rule either way: one pass over
        // one whole utterance.
        if let Some(acc) = accurate.as_mut()
            && args.mode == LaneMode::AccurateOnly
        {
            ring.push(&frame.pcm);
            if vad_events.contains(&VadEvent::SpeechStart) && speech_start.is_none() {
                speech_start = Some(now.saturating_sub(vad.silence()));
            }
            if pause && let Some(a) = speech_start.take() {
                let prompt = stream.prompt_tail();
                let (hyp, took) = segment(acc.as_mut(), &ring, a, now, &prompt, debug).await?;
                load.observe(now.saturating_sub(a), took);
                pass_ms.push(took.as_secs_f64() * 1e3);
                collect(stream.accurate_segment(&hyp, now), &mut lines, &mut sink)?;
            } else if speech_start.is_some_and(|a| now.saturating_sub(a) >= max_segment) {
                // A speaker who does not pause still has to reach the screen.
                let a = speech_start.replace(now).expect("checked above");
                let prompt = stream.prompt_tail();
                let (hyp, took) = segment(acc.as_mut(), &ring, a, now, &prompt, debug).await?;
                load.observe(now.saturating_sub(a), took);
                pass_ms.push(took.as_secs_f64() * 1e3);
                forced_flushes += 1;
                collect(stream.accurate_segment(&hyp, now), &mut lines, &mut sink)?;
            }
        }

        collect(stream.tick(now), &mut lines, &mut sink)?;
    }

    let end = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
    if let Some(fast) = fast.as_mut()
        && let Some(AsrEvent::Final {
            words,
            t_start,
            t_end,
            ..
        }) = fast.finalize().await?
    {
        fast_screen.push(t0.elapsed().as_secs_f64() - t_end.as_secs_f64());
        let words = match punct.as_mut() {
            Some(p) => p.restore_words(&words)?,
            None => words,
        };
        if let Some(policy) = cut.as_mut() {
            policy.settle(end);
            early_s.extend(
                open_cuts
                    .drain(..)
                    .map(|c| (end.saturating_sub(c)).as_secs_f64()),
            );
        }
        collect(stream.fast_final(&words, end), &mut lines, &mut sink)?;
        // The clip ended mid-utterance. Without this the last thing anyone said
        // never reaches the accurate lane at all, and the transcript ends on
        // whatever the fast lane heard -- a hole by any other name.
        if let Some(acc) = accurate.as_mut() {
            let prompt = stream.prompt_tail();
            let (hyp, took) =
                segment(acc.as_mut(), &ring, t_start, t_end.min(end), &prompt, debug).await?;
            load.observe(t_end.saturating_sub(t_start), took);
            pass_ms.push(took.as_secs_f64() * 1e3);
            collect(stream.accurate_segment(&hyp, end), &mut lines, &mut sink)?;
        }
    }
    if let Some(acc) = accurate.as_mut()
        && let Some(a) = speech_start.take()
    {
        let prompt = stream.prompt_tail();
        let (hyp, took) = segment(acc.as_mut(), &ring, a, end, &prompt, debug).await?;
        pass_ms.push(took.as_secs_f64() * 1e3);
        collect(stream.accurate_segment(&hyp, end), &mut lines, &mut sink)?;
    }
    // Nothing more is coming, so anything still waiting is promoted now rather
    // than being silently dropped from the transcript.
    collect(
        stream.tick(end + Duration::from_secs_f64(cfg.promote_after_s)),
        &mut lines,
        &mut sink,
    )?;
    let transcripts: Vec<PathBuf> = match sink.as_mut() {
        Some(w) => {
            let paths = w.paths().to_vec();
            w.finish()?;
            paths
        }
        None => Vec::new(),
    };

    let wall = t0.elapsed().as_secs_f64();
    let audio_s = samples as f64 / SAMPLE_RATE;
    let hypothesis = lines
        .iter()
        .map(|l| l.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let ref_path = args
        .reference
        .clone()
        .unwrap_or_else(|| reference_for(&args.wav));
    let reference = std::fs::read_to_string(&ref_path)
        .with_context(|| format!("reading the reference transcript {}", ref_path.display()))?;
    let m = crate::metrics::measure(&reference, &hypothesis, 5);

    let mut out = String::new();
    let name = args.wav.file_name().unwrap_or_default().to_string_lossy();
    out.push_str(&format!("# xtask eval — {name}\n\n"));
    for p in &transcripts {
        out.push_str(&format!("- transcript: `{}`\n", p.display()));
    }
    out.push_str(&format!(
        "- lanes: **{}**\n",
        match args.mode {
            LaneMode::Dual => "dual",
            LaneMode::FastOnly => "fast only",
            LaneMode::AccurateOnly => "accurate only",
        }
    ));
    for b in &backends {
        out.push_str(&format!("  - {b}\n"));
    }
    if let Some(pn) = punct.as_ref() {
        // The invariant the whole of stage 1 rests on: the model may add marks
        // and casing, never a word. Anything but 0 means some lines went
        // through unpunctuated, and every number below is then a mixture.
        out.push_str(&format!(
            "  - punctuation: {} (**word-count mismatches: {}**)\n",
            pn.model(),
            pn.mismatches()
        ));
    }
    out.push_str(&format!(
        "- audio {audio_s:.1}s, wall {wall:.1}s ({:+.1}%) | lines {}\n",
        (wall / audio_s - 1.0) * 100.0,
        lines.len()
    ));
    if let Some(rss) = peak_rss_mib() {
        out.push_str(&format!(
            "- **peak RSS: {rss:.0} MiB** (gate G4: <= 3584 MiB)\n"
        ));
    }
    if !load.is_empty() {
        out.push_str(&format!(
            "- accurate lane: {} passes, **{:.3} engine seconds per audio second** \
             (1.0 = exactly keeping up), mean utterance {:.1}s\n",
            pass_ms.len(),
            load.work_factor(),
            load.mean_window().as_secs_f64(),
        ));
    }
    if forced_flushes > 0 {
        out.push_str(&format!(
            "- {forced_flushes} utterances were cut at the {:.0}s limit rather than at a pause\n",
            cfg.max_segment_s
        ));
    }
    if args.cut {
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v.first().map(|_| v[v.len() / 2]).unwrap_or(0.0)
        };
        out.push_str(&format!(
            "- **semantic cut: {cuts} lines ended at a restored full stop**, \
             a median **{:.2}s of audio earlier** than the endpoint detector then \
             closed the same utterance; {cut_declined} candidates declined because \
             the decoded and punctuated views of the line disagreed about the words\n",
            med(&mut early_s),
        ));
    }

    out.push_str("\n| latency (s) | n | median | p90 | max |\n|---|---|---|---|---|\n");
    out.push_str(&row("G1 fast lane on screen", &fast_screen));
    let acc_lat: Vec<f64> = lines
        .iter()
        .filter(|l| l.lane == Lane::Accurate)
        .map(|l| l.latency)
        .collect();
    out.push_str(&row("G3 accurate lane overwrite", &acc_lat));
    out.push_str(&row(
        "line finalised (either lane)",
        &lines.iter().map(|l| l.latency).collect::<Vec<_>>(),
    ));

    let count = |r: Option<FastReason>| lines.iter().filter(|l| l.reason == r).count();
    out.push_str("\n| line came from | n |\n|---|---|\n");
    out.push_str(&format!("| accurate lane | {} |\n", count(None)));
    out.push_str(&format!(
        "| fast lane — accurate never answered | {} |\n",
        count(Some(FastReason::AccurateTimeout))
    ));
    out.push_str(&format!(
        "| fast lane — **accurate dropped a clause** | {} |\n",
        count(Some(FastReason::AccurateTruncated))
    ));
    out.push_str(&format!(
        "| fast lane — **accurate looped** | {} |\n",
        count(Some(FastReason::AccurateLooped))
    ));
    out.push_str(&format!(
        "| fast lane — accurate lane off | {} |\n",
        count(Some(FastReason::AccurateDisabled))
    ));

    let mut g1 = fast_screen.clone();
    g1.sort_by(f64::total_cmp);
    let mut g3 = acc_lat.clone();
    g3.sort_by(f64::total_cmp);
    let verdict = |ok: bool| if ok { "✅ PASS" } else { "❌ FAIL" };
    out.push_str(&format!(
        "\n| gate | requirement | measured | |\n|---|---|---|---|\n\
         | G1 | median <= 1.2s, p90 <= 2.0s | {:.2} / {:.2} | {} |\n\
         | G2a | content WER <= 15% | {:.1}% | {} |\n\
         | G2b | drop rate <= 2% | {:.1}% | {} |\n\
         | G3 | overwrite <= 4s (observed) | {:.2} | {} |\n",
        pct(&g1, 0.5),
        pct(&g1, 0.9),
        if fast_screen.is_empty() {
            "– n/a"
        } else {
            verdict(pct(&g1, 0.5) <= 1.2 && pct(&g1, 0.9) <= 2.0)
        },
        m.content_wer,
        verdict(m.content_wer <= 15.0),
        m.drop_rate,
        verdict(m.drop_rate <= 2.0),
        pct(&g3, 0.5),
        if acc_lat.is_empty() {
            "– n/a"
        } else {
            verdict(pct(&g3, 0.5) <= 4.0)
        },
    ));
    out.push_str(&format!(
        "\n- plain WER {:.1}% | content WER {:.1}% | drop rate {:.1}% over {} reference words\n",
        m.wer, m.content_wer, m.drop_rate, m.ref_words
    ));
    for d in &m.drops {
        out.push_str(&format!("  - **hole:** `{d}`\n"));
    }

    out.push_str("\n## Transcript\n\n");
    for l in &lines {
        let tag = match (l.lane, l.reason) {
            (Lane::Accurate, _) => String::new(),
            (Lane::Fast, Some(r)) => format!(" _[fast: {r:?}]_"),
            (Lane::Fast, None) => " _[fast]_".into(),
        };
        out.push_str(&format!("{}{tag}\n", l.text));
    }

    println!("{out}");
    if let Some(p) = &args.out {
        std::fs::write(p, &out)?;
        println!("[written to {}]", p.display());
    }
    Ok(())
}
