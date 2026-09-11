//! Linux capture through the PulseAudio API (PLAN §2.1).
//!
//! cpal cannot do this job on Linux. Its only Linux hosts are ALSA and JACK,
//! and **monitor sources do not exist at the ALSA layer** -- they are a
//! PulseAudio concept. On this machine ALSA offers cpal exactly two devices,
//! `default` and `pipewire`, while the Pulse layer lists eleven sources
//! including a `.monitor` for every sink. Capturing "whatever the machine is
//! playing" is a headline requirement, so Linux goes through Pulse instead.
//!
//! This costs nothing in portability: PipeWire ships `pipewire-pulse` and is
//! what every current desktop runs, and the same code works on a legacy
//! PulseAudio system. cpal stays for Windows, where WASAPI loopback is real.
//!
//! The server is asked for 16 kHz mono f32 directly and does the conversion,
//! which is both better and cheaper than doing it here -- the same arrangement
//! the Phase 0 PoC validated through `parec`.

use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use li_types::{AudioFrame, DeviceInfo, DeviceSelector};
use libpulse_binding::{
    context::{Context, FlagSet as ContextFlagSet, introspect::SourceInfo},
    def::BufferAttr,
    def::Retval,
    mainloop::standard::{IterateResult, Mainloop},
    sample::{Format, Spec},
    stream::Direction,
};
use libpulse_simple_binding::Simple;
use tokio::sync::mpsc::{self, Receiver};

use crate::{AudioSource, FRAME_MS, TARGET_RATE, agc::Agc};

const APP: &str = "LiveInterpreter";

#[derive(Default)]
pub struct PulseSource {
    /// Dropping this ends the capture thread.
    stop: Option<std::sync::mpsc::Sender<()>>,
    agc: bool,
}

impl PulseSource {
    pub fn new() -> Self {
        Self {
            stop: None,
            agc: true,
        }
    }

    pub fn with_agc(mut self, on: bool) -> Self {
        self.agc = on;
        self
    }
}

fn spec() -> Spec {
    Spec {
        format: Format::F32le,
        channels: 1,
        rate: TARGET_RATE,
    }
}

/// Resolve a selector to a Pulse source name.
fn source_name(sel: &DeviceSelector) -> Result<Option<String>> {
    match sel {
        // `None` lets the server pick its default source.
        DeviceSelector::Microphone => Ok(None),
        DeviceSelector::Device(id) => Ok(Some(id.clone())),
        DeviceSelector::SystemLoopback => {
            let (sources, default_sink) = introspect()?;
            // Follow the sink the user is actually listening on, so "system
            // sound" tracks a switch to headphones or Bluetooth.
            if let Some(sink) = default_sink {
                let want = format!("{sink}.monitor");
                if sources.iter().any(|s| s.id == want) {
                    return Ok(Some(want));
                }
            }
            sources
                .into_iter()
                .find(|s| s.is_loopback)
                .map(|s| Some(s.id))
                .ok_or_else(|| anyhow!("no monitor source available"))
        }
    }
}

/// Run a short mainloop to list sources and find the default sink.
fn introspect() -> Result<(Vec<DeviceInfo>, Option<String>)> {
    let mut main = Mainloop::new().ok_or_else(|| anyhow!("pulse: no mainloop"))?;
    let mut ctx = Context::new(&main, APP).ok_or_else(|| anyhow!("pulse: no context"))?;
    ctx.connect(None, ContextFlagSet::NOFLAGS, None)
        .context("connecting to the PulseAudio/PipeWire server")?;

    // Wait for the connection to settle before asking it anything.
    loop {
        match main.iterate(true) {
            IterateResult::Err(e) => bail!("pulse mainloop: {e}"),
            IterateResult::Quit(_) => bail!("pulse mainloop quit during connect"),
            IterateResult::Success(_) => {}
        }
        use libpulse_binding::context::State;
        match ctx.get_state() {
            State::Ready => break,
            State::Failed | State::Terminated => bail!("pulse: connection failed"),
            _ => {}
        }
    }

    let sources = Arc::new(Mutex::new(Vec::new()));
    let default_sink = Arc::new(Mutex::new(None));
    let done = Arc::new(Mutex::new(0usize));

    let (s, d) = (sources.clone(), done.clone());
    let op1 = ctx.introspect().get_source_info_list(move |res| match res {
        libpulse_binding::callbacks::ListResult::Item(SourceInfo {
            name, description, ..
        }) => {
            if let Some(name) = name.as_ref().map(|n| n.to_string()) {
                s.lock().unwrap().push(DeviceInfo {
                    is_loopback: name.ends_with(".monitor"),
                    name: description
                        .as_ref()
                        .map(|d| d.to_string())
                        .unwrap_or_else(|| name.clone()),
                    id: name,
                });
            }
        }
        libpulse_binding::callbacks::ListResult::End
        | libpulse_binding::callbacks::ListResult::Error => *d.lock().unwrap() += 1,
    });

    let (k, d2) = (default_sink.clone(), done.clone());
    let op2 = ctx.introspect().get_server_info(move |info| {
        *k.lock().unwrap() = info.default_sink_name.as_ref().map(|n| n.to_string());
        *d2.lock().unwrap() += 1;
    });

    while *done.lock().unwrap() < 2 {
        match main.iterate(true) {
            IterateResult::Err(e) => bail!("pulse mainloop: {e}"),
            IterateResult::Quit(_) => break,
            IterateResult::Success(_) => {}
        }
    }
    drop((op1, op2));
    ctx.disconnect();
    main.quit(Retval(0));

    let sources = std::mem::take(&mut *sources.lock().unwrap());
    let sink = default_sink.lock().unwrap().clone();
    Ok((sources, sink))
}

#[async_trait]
impl AudioSource for PulseSource {
    fn list_devices() -> Result<Vec<DeviceInfo>> {
        Ok(introspect()?.0)
    }

    async fn open(&mut self, sel: DeviceSelector) -> Result<Receiver<AudioFrame>> {
        let name = source_name(&sel)?;
        let frame_len = (TARGET_RATE as usize * FRAME_MS as usize) / 1000;
        let (tx, rx) = mpsc::channel(64);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let agc_on = self.agc;

        // `Simple` is a blocking read API, so it gets a thread. That is the
        // right shape anyway: PLAN §11 wants capture doing nothing but filling
        // a buffer, with no inference or I/O on the same path.
        std::thread::spawn(move || {
            // Ask for one frame at a time. Left to itself the server picks the
            // fragment size, and on a monitor of a *Bluetooth* sink that is
            // sized for the link's own buffering, not for a subtitle bar --
            // audio then arrives in lumps, and every lump is latency that
            // nothing downstream can give back. `fragsize` is the only one of
            // these fields a record stream reads; the rest are playback's and
            // are left at "server decides".
            let attr = BufferAttr {
                maxlength: u32::MAX,
                tlength: u32::MAX,
                prebuf: u32::MAX,
                minreq: u32::MAX,
                fragsize: (frame_len * 4) as u32,
            };
            let simple = Simple::new(
                None,
                APP,
                Direction::Record,
                name.as_deref(),
                "subtitles",
                &spec(),
                None,
                Some(&attr),
            );
            let simple = match simple {
                Ok(s) => {
                    let _ = ready_tx.send(Ok(()));
                    s
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow!("opening pulse source {name:?}: {e}")));
                    return;
                }
            };
            tracing::info!(source = ?name, fragment_ms = FRAME_MS, "capture open");

            let mut agc = Agc::default();
            // The server's own latency: how old the audio in a read already is
            // before this program has seen a sample of it. Everything the bar
            // shows is at least this late and nothing downstream can give it
            // back, so it is the first number to look for when someone reports
            // that subtitles feel slow -- a monitor of a *Bluetooth* sink
            // carries the link's buffering and reports far more than a built-in
            // one. Read on the running stream, because before the first read
            // there is nothing to report and it answers 0.
            let mut latency_reported: Option<std::time::Instant> = None;
            let mut bytes = vec![0u8; frame_len * 4];
            // Log the *edges* of a stall, not every dropped frame: at 31 frames
            // a second, per-frame warnings would themselves cost latency.
            let mut dropping = false;
            loop {
                if stop_rx.try_recv() != Err(std::sync::mpsc::TryRecvError::Empty) {
                    return;
                }
                if let Err(e) = simple.read(&mut bytes) {
                    tracing::error!("capture read failed: {e}");
                    return;
                }
                let mut pcm: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                if agc_on {
                    agc.process(&mut pcm);
                }
                if latency_reported.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(5))
                {
                    latency_reported = Some(std::time::Instant::now());
                    if let Ok(l) = simple.get_latency() {
                        tracing::info!(ms = l.0 / 1000, "audio server latency");
                    }
                }
                // Bounded: drop old audio rather than grow a queue (§11).
                // Losing a frame costs a word; an unbounded queue costs the gate.
                match tx.try_send(AudioFrame {
                    pcm,
                    sample_rate: TARGET_RATE,
                    t_capture: std::time::Instant::now(),
                }) {
                    Err(mpsc::error::TrySendError::Closed(_)) => return,
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        if !dropping {
                            dropping = true;
                            tracing::warn!("pipeline behind; dropping frames");
                        }
                    }
                    Ok(()) => {
                        if dropping {
                            dropping = false;
                            tracing::info!("pipeline caught up");
                        }
                    }
                }
            }
        });

        ready_rx
            .await
            .context("capture thread died during setup")??;
        self.stop = Some(stop_tx);
        Ok(rx)
    }
}
