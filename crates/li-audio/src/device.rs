//! Desktop capture through cpal.
//!
//! This is the Windows path, and the fallback for any platform that is neither
//! Windows nor Linux -- Linux goes through [`crate::pulse`] instead, because
//! monitor sources do not exist below the Pulse layer.
//!
//! "System sound" is a different thing on each platform, and cpal does not
//! paper over the difference:
//!
//! * **Windows/WASAPI has no loopback device.** Loopback is a *mode of the
//!   output endpoint*: opening a render endpoint for input makes WASAPI hand
//!   back what is being played. cpal does exactly this -- it sets
//!   `AUDCLNT_STREAMFLAGS_LOOPBACK` when the device it is told to build an
//!   input stream on turns out to be a render endpoint -- so "capture the
//!   system sound" here means "find the right *output* device and build an
//!   input stream on it". Two consequences this module has to carry:
//!   `default_input_config` refuses such a device outright (`eRender` has no
//!   input formats), so the format has to be asked for on the output side; and
//!   an idle endpoint delivers nothing at all rather than delivering silence,
//!   which is what [`Silence`] is for.
//! * **Linux/PipeWire** exposes a monitor source per sink, which shows up in
//!   the ordinary input list with a `.monitor` suffix -- a naming problem
//!   rather than an API one.
//!
//! Everything else -- USB interfaces, Bluetooth, a second sound card -- is an
//! ordinary input on every platform and needs no special case.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use li_types::{AudioFrame, DeviceInfo, DeviceSelector};
use tokio::sync::mpsc::{self, Receiver};

use crate::{AudioSource, FRAME_MS, TARGET_RATE, agc::Agc, resample::Resampler};

/// PipeWire and PulseAudio both name monitor sources this way.
#[cfg(not(windows))]
const MONITOR_SUFFIX: &str = ".monitor";

/// Which endpoint list a device came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Input,
    /// A render endpoint opened for input: Windows loopback. Never constructed
    /// on other platforms, where the loopback tap is an ordinary input device.
    #[cfg_attr(not(windows), allow(dead_code))]
    Output,
}

pub struct DeviceSource {
    /// Dropping this ends the thread that owns the stream, which stops capture.
    stop: Option<std::sync::mpsc::Sender<()>>,
    agc: bool,
}

impl Default for DeviceSource {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceSource {
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

#[cfg(windows)]
fn pick(sel: &DeviceSelector) -> Result<(cpal::Device, Side)> {
    let host = cpal::default_host();
    match sel {
        DeviceSelector::Microphone => host
            .default_input_device()
            .map(|d| (d, Side::Input))
            .ok_or_else(|| anyhow!("no default input device")),
        // Follow the *default* output endpoint, so "system sound" tracks a
        // switch to headphones or a Bluetooth speaker -- the same rule the
        // Pulse path applies to the default sink's monitor.
        DeviceSelector::SystemLoopback => host
            .default_output_device()
            .map(|d| (d, Side::Output))
            .ok_or_else(|| anyhow!("no default output device to capture from")),
        DeviceSelector::Device(id) => by_name(&host, id),
    }
}

/// A named endpoint from either list.
///
/// Inputs are searched first, so a name that exists on both sides resolves to
/// the real input. On WASAPI the name *is* the identifier as far as cpal is
/// concerned, so two identical devices plugged in at once cannot be told apart
/// here; that is a cpal limitation rather than a choice, and it is why the id
/// written to `config.toml` is a name.
#[cfg(windows)]
fn by_name(host: &cpal::Host, id: &str) -> Result<(cpal::Device, Side)> {
    let named = |d: &cpal::Device| d.name().is_ok_and(|n| n == id);
    if let Some(d) = host.input_devices()?.find(named) {
        return Ok((d, Side::Input));
    }
    host.output_devices()?
        .find(named)
        .map(|d| (d, Side::Output))
        .with_context(|| format!("no audio device named {id:?}"))
}

#[cfg(not(windows))]
fn pick(sel: &DeviceSelector) -> Result<(cpal::Device, Side)> {
    let host = cpal::default_host();
    let device = match sel {
        DeviceSelector::Microphone => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?,
        DeviceSelector::SystemLoopback => {
            // Prefer the monitor of the *default* sink, so "system sound"
            // follows whatever the user is actually listening on.
            let want = host
                .default_output_device()
                .and_then(|d| d.name().ok())
                .map(|n| format!("{n}{MONITOR_SUFFIX}"));
            let mut monitors = host
                .input_devices()?
                .filter(|d| d.name().is_ok_and(|n| n.ends_with(MONITOR_SUFFIX)));
            if let Some(want) = want
                && let Some(d) = host
                    .input_devices()?
                    .find(|d| d.name().is_ok_and(|n| n == want))
            {
                return Ok((d, Side::Input));
            }
            monitors.next().ok_or_else(|| {
                anyhow!(
                    "no monitor source found; on Linux this needs PipeWire or PulseAudio, \
                     on Windows use the loopback device"
                )
            })?
        }
        DeviceSelector::Device(id) => host
            .input_devices()?
            .find(|d| d.name().is_ok_and(|n| &n == id))
            .with_context(|| format!("no input device named {id:?}"))?,
    };
    Ok((device, Side::Input))
}

/// The inputs, plus -- on Windows -- every output endpoint, because each one is
/// a capturable loopback there.
#[cfg(windows)]
fn list() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let named = |is_loopback: bool| {
        move |name: String| DeviceInfo {
            is_loopback,
            id: name.clone(),
            name,
        }
    };
    let mut out: Vec<DeviceInfo> = host
        .input_devices()?
        .filter_map(|d| d.name().ok())
        .map(named(false))
        .collect();
    // Marked, not renamed. A Windows output endpoint is called "Speakers
    // (Realtek(R) Audio)" whether you mean to play through it or record from
    // it, so the flag is the only thing that tells the settings window which
    // half of the list it is looking at.
    out.extend(
        host.output_devices()?
            .filter_map(|d| d.name().ok())
            .map(named(true)),
    );
    Ok(out)
}

#[cfg(not(windows))]
fn list() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    Ok(host
        .input_devices()?
        .filter_map(|d| d.name().ok())
        .map(|name| DeviceInfo {
            is_loopback: name.ends_with(MONITOR_SUFFIX),
            id: name.clone(),
            name,
        })
        .collect())
}

/// Shorter than any real stall, longer than any scheduling jitter between two
/// WASAPI packets (which arrive about every 10 ms).
const IDLE_GAP: Duration = Duration::from_millis(250);
/// The longest gap worth rebuilding. Past a couple of seconds nothing
/// downstream is still waiting on it: the sentence has already been flushed by
/// the accurate lane's own time limit, and the frames would go into a bounded
/// channel that drops them. What is left to pay is the timestamp column, the
/// cheapest of the three things a gap costs.
const MAX_FILL: Duration = Duration::from_secs(2);

/// Rebuilds the silence a loopback endpoint does not deliver.
///
/// A WASAPI render endpoint with nothing playing through it produces no
/// packets at all -- the capture callback is simply not called until sound
/// resumes. Downstream, audio time is a sample count (`li-core`'s pipeline
/// clock), so an unreconstructed gap does not read as a quiet stretch: it
/// reads as if the audio on either side of it were adjacent. Two things break
/// at once. The endpointer never sees the trailing silence that ends a
/// sentence, so the last line sits unfinalised until somebody speaks again;
/// and every timestamp after the gap is early by the length of the gap.
///
/// So the gap is measured against the wall clock and filled with zeros. Only a
/// real stall counts, which is the point of [`IDLE_GAP`]: the estimate stays
/// local to one callback, so the device clock's drift against the system clock
/// cannot accumulate into silence inserted in the middle of speech.
#[derive(Debug, Default)]
struct Silence {
    last: Option<Instant>,
}

impl Silence {
    /// How much silence to insert before a callback that arrived at `now`
    /// carrying `covered` of audio.
    fn missing(&mut self, now: Instant, covered: Duration) -> Duration {
        // The first callback has nothing to be late for: capture starts here.
        let Some(last) = self.last.replace(now) else {
            return Duration::ZERO;
        };
        let gap = now.saturating_duration_since(last).saturating_sub(covered);
        if gap < IDLE_GAP {
            Duration::ZERO
        } else {
            gap.min(MAX_FILL)
        }
    }
}

#[async_trait]
impl AudioSource for DeviceSource {
    fn list_devices() -> Result<Vec<DeviceInfo>> {
        list()
    }

    async fn open(&mut self, sel: DeviceSelector) -> Result<Receiver<AudioFrame>> {
        // `cpal::Stream` is not `Send` on every backend, so instead of reaching
        // for `unsafe impl Send` we give it a thread of its own: the thread
        // builds it, starts it, and then just holds it alive until `stop` is
        // dropped. Nothing else ever touches the handle.
        let (tx, rx) = mpsc::channel(64);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let agc_on = self.agc;

        std::thread::spawn(move || match build(sel, agc_on, tx) {
            Ok(stream) => {
                let _ = ready_tx.send(Ok(()));
                let _ = stop_rx.recv(); // parks until the source is dropped
                drop(stream);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        });

        ready_rx
            .await
            .context("capture thread died during setup")??;
        self.stop = Some(stop_tx);
        Ok(rx)
    }
}

/// Build and start the capture stream. Runs on the stream's own thread.
fn build(sel: DeviceSelector, agc_on: bool, tx: mpsc::Sender<AudioFrame>) -> Result<cpal::Stream> {
    let (device, side) = pick(&sel)?;
    let config = match side {
        Side::Input => device.default_input_config(),
        // A loopback capture is shaped like what the endpoint plays, so the
        // format question goes to the output side. Shared mode gives no choice
        // about it anyway: the endpoint's mix format is the format.
        Side::Output => device.default_output_config(),
    }
    .context("device has no usable input config")?;
    let sample_format = config.sample_format();
    let config: cpal::StreamConfig = config.into();
    let channels = config.channels;
    let rate = config.sample_rate.0;
    tracing::info!(
        device = device.name().unwrap_or_default(),
        rate,
        channels,
        ?sample_format,
        loopback = side == Side::Output,
        "capture open"
    );

    let mut resampler = Resampler::new(rate, TARGET_RATE);
    let mut agc = Agc::default();
    let frame_len = (TARGET_RATE as usize * FRAME_MS as usize) / 1000;
    let mut out: Vec<f32> = Vec::with_capacity(frame_len * 4);
    // Only the loopback path. A real capture device keeps delivering when the
    // room is quiet, so a gap there means something else went wrong, and
    // papering over it would hide it.
    let mut idle = (side == Side::Output).then(Silence::default);

    let mut dropping = false;
    // The size of the device's own buffer, reported once.
    //
    // It is latency nothing downstream can see or undo: the audio in the first
    // callback happened up to this long before the callback ran, and every
    // clock in the engine starts from the callback. Whatever the bar shows is
    // at least this late. cpal is left on `BufferSize::Default` deliberately --
    // asking for a smaller one risks xruns on a device we have not measured --
    // so the number has to come out of the log instead.
    let mut announced = false;
    // The callback does only arithmetic and a non-blocking send. No I/O, no
    // inference, no lock that another thread can hold.
    let mut feed = move |input: &[f32]| {
        if !announced {
            announced = true;
            let frames = input.len() / channels.max(1) as usize;
            tracing::info!(
                frames,
                ms = (frames as f64 * 1000.0 / f64::from(rate)).round() as u64,
                "capture buffer"
            );
        }
        if let Some(idle) = idle.as_mut() {
            let covered =
                Duration::from_secs_f64(input.len() as f64 / channels as f64 / rate as f64);
            let missing = idle.missing(Instant::now(), covered);
            if !missing.is_zero() {
                tracing::debug!(ms = missing.as_millis(), "loopback idle; inserting silence");
                // Straight into the resampled buffer: silence resamples to
                // silence, and going around the filter leaves its state where
                // the audio left it.
                let n = (missing.as_secs_f64() * f64::from(TARGET_RATE)) as usize;
                out.extend(std::iter::repeat_n(0.0, n));
            }
        }
        out.extend(resampler.process(input, channels));
        while out.len() >= frame_len {
            let mut pcm: Vec<f32> = out.drain(..frame_len).collect();
            if agc_on {
                agc.process(&mut pcm);
            }
            // Bounded channel: if the pipeline stalls, drop the oldest audio
            // rather than grow a queue. Losing a frame costs a word; an
            // unbounded queue costs the latency gate.
            match tx.try_send(AudioFrame {
                pcm,
                sample_rate: TARGET_RATE,
                t_capture: std::time::Instant::now(),
            }) {
                // Log the edges of a stall, not every dropped frame: at 31
                // frames a second, per-frame warnings would themselves cost
                // latency.
                Err(mpsc::error::TrySendError::Full(_)) if !dropping => {
                    dropping = true;
                    tracing::warn!("pipeline behind; dropping frames");
                }
                Ok(()) if dropping => {
                    dropping = false;
                    tracing::info!("pipeline caught up");
                }
                _ => {}
            }
        }
    };

    let err = |e| tracing::error!("capture stream error: {e}");
    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            device.build_input_stream(&config, move |d: &[f32], _| feed(d), err, None)
        }
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |d: &[i16], _| {
                let f: Vec<f32> = d.iter().map(|&s| s as f32 / 32768.0).collect();
                feed(&f)
            },
            err,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |d: &[u16], _| {
                let f: Vec<f32> = d.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).collect();
                feed(&f)
            },
            err,
            None,
        ),
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    }
    .context("building the capture stream")?;

    stream.play().context("starting the capture stream")?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 480 stereo samples at 48 kHz: one 10 ms WASAPI packet.
    const PACKET: Duration = Duration::from_millis(10);

    #[test]
    fn a_stream_that_keeps_up_gets_nothing_inserted() {
        let mut s = Silence::default();
        let t0 = Instant::now();
        for i in 0..200 {
            let at = t0 + PACKET * i;
            assert_eq!(s.missing(at, PACKET), Duration::ZERO);
        }
    }

    #[test]
    fn scheduling_jitter_is_not_a_gap() {
        // Packets arriving late but not stalled: WASAPI delivers what it owes,
        // so filling here would insert silence into continuous audio.
        let mut s = Silence::default();
        let t0 = Instant::now();
        s.missing(t0, PACKET);
        for i in 1..50 {
            let at = t0 + PACKET * i + Duration::from_millis(80);
            assert_eq!(s.missing(at, PACKET), Duration::ZERO);
        }
    }

    #[test]
    fn an_idle_endpoint_gets_its_silence_back() {
        let mut s = Silence::default();
        let t0 = Instant::now();
        s.missing(t0, PACKET);
        // Nothing played for 1.5 s; the next packet carries 10 ms of audio.
        let gap = Duration::from_millis(1500);
        assert_eq!(s.missing(t0 + gap, PACKET), gap - PACKET);
    }

    #[test]
    fn a_long_idle_is_filled_only_up_to_the_cap() {
        let mut s = Silence::default();
        let t0 = Instant::now();
        s.missing(t0, PACKET);
        assert_eq!(
            s.missing(t0 + Duration::from_secs(600), PACKET),
            MAX_FILL,
            "ten minutes of nothing must not become ten minutes of frames"
        );
    }

    #[test]
    fn the_first_callback_is_never_late() {
        // Capture starts at the first packet. Measuring against `Instant::now`
        // at construction would charge the stream for the device's own start-up.
        let mut s = Silence::default();
        assert_eq!(
            s.missing(Instant::now() + Duration::from_secs(30), PACKET),
            Duration::ZERO
        );
    }
}
