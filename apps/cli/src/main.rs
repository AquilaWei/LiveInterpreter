//! `liveinterpreter` — the whole pipeline on a terminal.
//!
//! This is the alpha checkpoint: the first build that can be used rather than
//! measured. It is also the debugging tool for everything after it — when the
//! floating bar shows nothing, this says whether the engine or the UI is at
//! fault.
//!
//! ```text
//! liveinterpreter --list-devices
//! liveinterpreter                      # whatever the machine is playing
//! liveinterpreter --source mic
//! liveinterpreter --lanes fast --no-mt # what the fast lane alone hears
//! ```
//!
//! Logs go to stderr and the subtitle to stdout, so `2>/dev/null` gives a clean
//! bar and `2>log` keeps the warnings.

mod bar;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use li_audio::{AudioSource, DesktopSource};
use li_core::Engine;
use li_core::config::EngineConfig;
use li_core::download;
use li_core::models::Models;
use li_types::{DeviceSelector, EngineEvent, EngineStatus};

#[derive(Debug)]
struct Args {
    source: Option<DeviceSelector>,
    lanes: Option<String>,
    no_mt: bool,
    transcript_dir: Option<PathBuf>,
    no_transcript: bool,
    device: Option<String>,
    config: Option<PathBuf>,
    wav: Option<PathBuf>,
    list_devices: bool,
    fetch_models: bool,
}

const USAGE: &str = "\
usage: liveinterpreter [options]

  --list-devices          list capture devices and exit
  --fetch-models          download the models this config needs, then exit
  --source S              loopback (default) | mic | <device name>
  --lanes L               dual (default) | fast | accurate
  --device D              auto | cpu | vulkan | cuda | sycl   (accurate lane)
  --no-mt                 source subtitles only
  --transcript-dir DIR    where the transcript files go
  --no-transcript         do not write any transcript
  --config FILE           use this config.toml
  --wav FILE              play a wav file instead of listening (for checking)
  --version               print the version and the commit it was built from
";

/// What this build is, for a bug report and for the startup log.
///
/// The version comes from `Cargo.toml`; the rest from `build.rs`, because the
/// version alone has never been enough -- two packages a week apart both said
/// `0.1.0`, and only the commit says which one is installed.
const BUILD: &str = concat!(
    "LiveInterpreter ",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("LI_BUILD_ID"),
    ")"
);

fn parse(mut it: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args {
        source: None,
        lanes: None,
        no_mt: false,
        transcript_dir: None,
        no_transcript: false,
        fetch_models: false,
        device: None,
        config: None,
        wav: None,
        list_devices: false,
    };
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{flag} is missing its value"))
        };
        match flag.as_str() {
            "--list-devices" => a.list_devices = true,
            "--fetch-models" => a.fetch_models = true,
            "--no-mt" => a.no_mt = true,
            "--no-transcript" => a.no_transcript = true,
            "--source" => a.source = Some(DeviceSelector::from_name(&val()?)),
            "--lanes" => a.lanes = Some(val()?),
            "--device" => a.device = Some(val()?),
            "--transcript-dir" => a.transcript_dir = Some(val()?.into()),
            "--config" => a.config = Some(val()?.into()),
            "--wav" => a.wav = Some(val()?.into()),
            "--version" | "-V" => {
                println!("{BUILD}");
                std::process::exit(0);
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown flag {other}\n\n{USAGE}"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("LI_LOG")
                // `liveinterpreter` is this binary's own tracing target, and
                // without it the line saying which build this is -- the whole
                // point of writing it -- never reaches the log.
                .unwrap_or_else(|_| "warn,liveinterpreter=info,li_core=info,li_asr=info".into()),
        )
        .init();

    // First line in the log, always: every report that starts "it does X" has
    // to be attached to a build before it means anything.
    tracing::info!("{BUILD}");

    let args = parse(std::env::args().skip(1))?;
    if args.list_devices {
        return list_devices();
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    if args.fetch_models {
        return rt.block_on(fetch_models(&args));
    }
    rt.block_on(run(args))
}

/// Download whatever this config needs and stop.
///
/// Separate from `run` rather than folded into startup because it is the one
/// way to exercise the downloader without a display, a sound card, or a model
/// -- which is exactly the machine a fresh install is on.
async fn fetch_models(args: &Args) -> Result<()> {
    let cfg = resolve(args)?;
    let models = Models::new();
    let plan = download::plan(&models, &cfg)?;
    if plan.is_empty() {
        println!("All models are already in {}.", models.root().display());
        return Ok(());
    }
    println!(
        "Fetching {} into {}:",
        human(plan.total_bytes),
        models.root().display()
    );
    for m in plan.models() {
        println!("  {m}");
    }

    // Redrawn in place, so a thousand chunks do not become a thousand lines.
    // stderr because the transfer is progress, not output, and someone piping
    // this wants the two apart.
    let mut last = std::time::Instant::now();
    let mut current = String::new();
    download::fetch(&plan, |p| match p {
        download::Progress::Started { name, .. } => current = name,
        download::Progress::Bytes { done, total } => {
            // Every chunk is far more often than a terminal can be read.
            if last.elapsed() >= std::time::Duration::from_millis(200) {
                last = std::time::Instant::now();
                let pct = (done * 100).checked_div(total).unwrap_or(0);
                eprint!(
                    "\r  {pct:3}%  {} / {}  {current}\x1b[K",
                    human(done),
                    human(total)
                );
            }
        }
        download::Progress::Verifying { name } => {
            eprint!("\r  checking {name}\x1b[K");
        }
        download::Progress::Unpacking { name } => {
            eprint!("\r  unpacking {name}\x1b[K");
        }
        download::Progress::Finished { name } => {
            eprintln!("\r  done {name}\x1b[K");
        }
    })
    .await?;
    println!("All models are in {}.", models.root().display());
    Ok(())
}

fn human(bytes: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    let mb = bytes as f64 / MB;
    if mb >= 1024.0 {
        format!("{:.2} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

fn list_devices() -> Result<()> {
    for d in DesktopSource::list_devices().context("listing capture devices")? {
        let kind = if d.is_loopback {
            "loopback"
        } else {
            "input   "
        };
        println!("  [{kind}] {}", d.name);
    }
    println!("\n  --source loopback   whatever the machine is playing (the default)");
    println!("  --source mic        the default microphone");
    println!("  --source <name>     any name above");
    Ok(())
}

/// Apply the command line over the config file.
///
/// The file is the settings; the flags are for one run. Nothing here writes the
/// file back — that is the settings window's job.
fn resolve(args: &Args) -> Result<EngineConfig> {
    let mut cfg = match &args.config {
        Some(p) => EngineConfig::load_from(p)?,
        None => EngineConfig::load()?,
    };
    if let Some(sel) = &args.source {
        cfg.audio.source = sel.clone();
    }
    match args.lanes.as_deref() {
        None | Some("dual") => {}
        Some("fast") => cfg.asr.accurate = None,
        Some("accurate") => cfg.asr.fast = None,
        Some(other) => bail!("--lanes wants dual|fast|accurate, not {other}"),
    }
    if let Some(d) = &args.device
        && let Some(lane) = cfg.asr.accurate.as_mut()
    {
        lane.device = d.parse()?;
    }
    if args.no_mt {
        cfg.mt.backend = "off".into();
    }
    if let Some(dir) = &args.transcript_dir {
        cfg.transcript.dir = dir.clone();
    }
    if args.no_transcript {
        cfg.transcript.enabled = false;
    }
    Ok(cfg)
}

async fn run(args: Args) -> Result<()> {
    let cfg = resolve(&args)?;
    let mut engine = Engine::new(cfg)?;
    let mut events = engine.subscribe();

    let mut bar = bar::Bar::open()?;
    bar.set_status("starting…");
    bar.draw()?;

    // Subscribed before `start`, so the model-loading status reaches the bar.
    let started = match &args.wav {
        Some(wav) => engine.start_from_wav(wav).await,
        None => engine.start().await,
    };
    if let Err(e) = started {
        drop(bar);
        return Err(e);
    }
    for p in engine.transcripts() {
        eprintln!("transcript: {}", p.display());
    }

    let mut shown = Shown::default();
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            ev = events.recv() => match ev {
                // The engine ended on its own: the wav ran out, or the capture
                // device went away.
                Ok(EngineEvent::Status(EngineStatus::Stopped)) => break,
                Ok(ev) => {
                    show(&mut bar, &mut shown, ev);
                    bar.draw()?;
                }
                // The engine outran the terminal. Nothing to do but carry on
                // with the next event; the bar is replaced wholesale anyway.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("the display fell {n} events behind");
                }
                Err(_) => break,
            },
        }
    }

    bar.set_status("stopping…");
    bar.draw()?;
    engine.stop().await?;
    drop(bar);
    println!();
    Ok(())
}

/// The newest line each row has shown. Neither row ever goes backwards.
///
/// The accurate lane settles a line about two seconds after the fast lane
/// opened it, by which time the speaker is usually a sentence further on, so a
/// settled line routinely arrives after the row has moved on. Following
/// whichever line the last event named -- which is what this did until task
/// bar -- makes the display rewind to the previous sentence and then jump
/// forward again, several times a minute. A late line is dropped from the
/// screen instead; `li-transcript` already has it, and the file is where the
/// accurate text is meant to end up.
///
/// The floating bar follows the same rule, so the two can be
/// compared when one of them looks wrong.
#[derive(Default)]
struct Shown {
    source: Option<u64>,
    translation: Option<u64>,
}

impl Shown {
    fn take(slot: &mut Option<u64>, line_id: u64) -> bool {
        if slot.is_some_and(|shown| line_id < shown) {
            return false;
        }
        *slot = Some(line_id);
        true
    }
}

fn show(bar: &mut bar::Bar, shown: &mut Shown, ev: EngineEvent) {
    match ev {
        EngineEvent::SourcePartial { line_id, text, .. } => {
            if Shown::take(&mut shown.source, line_id) {
                bar.partial(&text);
            }
        }
        EngineEvent::SourceFinal {
            line_id,
            text,
            lane,
            reason,
            ..
        } => {
            if Shown::take(&mut shown.source, line_id) {
                bar.source_final(&text);
            }
            if let Some(r) = reason {
                tracing::info!(line_id, ?lane, ?r, "promoted from the fast lane");
            }
        }
        EngineEvent::Translation { line_id, text, .. } => {
            if Shown::take(&mut shown.translation, line_id) {
                bar.translation(&text);
            }
        }
        EngineEvent::Status(s) => bar.set_status(status(&s)),
        EngineEvent::Metrics(_) => {}
    }
}

fn status(s: &EngineStatus) -> String {
    match s {
        EngineStatus::ModelLoading { what } => format!("loading {what}…"),
        EngineStatus::Running => "listening — ctrl-c to stop".into(),
        EngineStatus::Paused => "paused".into(),
        EngineStatus::Reconnecting { backend } => format!("reconnecting to {backend}…"),
        EngineStatus::Error { message } => format!("error: {message}"),
        EngineStatus::Stopped => "stopped".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        parse(v.iter().map(|s| s.to_string())).unwrap()
    }

    /// An empty file and not the machine's own config: `resolve` falls back to
    /// `EngineConfig::load()`, so this test used to assert the built-in default
    /// while reading whatever the person running it had saved. It passed until
    /// someone set `source = "mic"` on their own machine, which is a test
    /// reporting the developer's settings rather than the program's.
    fn empty_config() -> std::path::PathBuf {
        let p = std::env::temp_dir().join("li_cli_empty_config.toml");
        std::fs::write(&p, "").unwrap();
        p
    }

    #[test]
    fn the_default_source_is_whatever_the_machine_is_playing() {
        let empty = empty_config();
        let cfg = resolve(&args(&[
            "--no-transcript",
            "--config",
            empty.to_str().unwrap(),
        ]))
        .unwrap();
        assert_eq!(cfg.audio.source, DeviceSelector::SystemLoopback);
    }

    #[test]
    fn a_named_device_is_taken_as_a_name() {
        let a = args(&[
            "--source",
            "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
        ]);
        assert!(matches!(a.source, Some(DeviceSelector::Device(_))));
        let a = args(&["--source", "mic"]);
        assert_eq!(a.source, Some(DeviceSelector::Microphone));
    }

    #[test]
    fn one_lane_flag_switches_the_other_lane_off() {
        let cfg = resolve(&args(&["--lanes", "fast", "--no-transcript"])).unwrap();
        assert!(cfg.asr.accurate.is_none() && cfg.asr.fast.is_some());
        let cfg = resolve(&args(&["--lanes", "accurate", "--no-transcript"])).unwrap();
        assert!(cfg.asr.fast.is_none() && cfg.asr.accurate.is_some());
    }

    #[test]
    fn an_unknown_lane_is_rejected_with_the_real_ones() {
        let err = resolve(&args(&["--lanes", "both"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("dual|fast|accurate"), "{err}");
    }

    #[test]
    fn no_mt_turns_the_translator_off_rather_than_hiding_it() {
        // The difference matters: a hidden translator still costs 0.8 GB.
        let cfg = resolve(&args(&["--no-mt", "--no-transcript"])).unwrap();
        assert_eq!(cfg.mt.backend, "off");
    }

    /// The failure this prevents: the accurate lane settles line 4 while line 5
    /// is already on screen, the display rewinds to 4, and the next fast-lane
    /// partial jumps it forward again.
    #[test]
    fn a_line_settled_late_does_not_rewind_the_display() {
        let mut shown = Shown::default();
        assert!(Shown::take(&mut shown.source, 4));
        assert!(Shown::take(&mut shown.source, 5));
        assert!(!Shown::take(&mut shown.source, 4));
        assert!(
            Shown::take(&mut shown.source, 5),
            "line 5's own accurate text still replaces it, in place"
        );
    }

    /// The two rows run on different clocks, so they advance separately: a
    /// translation for line 4 arriving while line 5 is on the source row is the
    /// normal case, not a late one.
    #[test]
    fn the_translation_row_keeps_its_own_place() {
        let mut shown = Shown::default();
        assert!(Shown::take(&mut shown.source, 5));
        assert!(Shown::take(&mut shown.translation, 4));
        assert_eq!((shown.source, shown.translation), (Some(5), Some(4)));
    }

    #[test]
    fn an_unknown_flag_prints_the_usage() {
        let err = parse(["--lanes-please".to_string()].into_iter())
            .unwrap_err()
            .to_string();
        assert!(err.contains("--list-devices"), "{err}");
    }
}
