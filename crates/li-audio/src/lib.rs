//! Audio capture: system loopback, microphone, or any other input device.
//!
//! Every source hands `li-core` the same thing: 16 kHz mono f32 frames with the
//! input level already normalised and the rate change properly filtered. Both of
//! those are here rather than in a backend because they are where the recogniser
//! quietly fails -- see [`agc`] and [`resample`].

pub mod agc;
pub mod device;
pub mod file;
#[cfg(target_os = "linux")]
pub mod pulse;
pub mod resample;

use anyhow::Result;
use async_trait::async_trait;
use li_types::{AudioFrame, DeviceInfo, DeviceSelector};
use tokio::sync::mpsc::Receiver;

/// What every model downstream expects.
pub const TARGET_RATE: u32 = 16_000;
/// Frame length. Short enough that the VAD reacts promptly, long enough that
/// the channel is not the bottleneck.
pub const FRAME_MS: u32 = 32;

/// The capture backend for this platform.
///
/// Linux goes through PulseAudio rather than cpal because monitor sources --
/// "whatever the machine is playing" -- do not exist at the ALSA layer that
/// cpal's Linux host talks to. See [`pulse`].
#[cfg(target_os = "linux")]
pub type DesktopSource = pulse::PulseSource;
#[cfg(not(target_os = "linux"))]
pub type DesktopSource = device::DeviceSource;

#[async_trait]
pub trait AudioSource: Send {
    fn list_devices() -> Result<Vec<DeviceInfo>>
    where
        Self: Sized;

    /// Start capturing and hand back the frame stream.
    ///
    /// Returning the receiver from `open` rather than a separate `subscribe`
    /// keeps it single-consumer by construction: these are frames, not events,
    /// and two readers would each get an arbitrary half of the audio.
    async fn open(&mut self, sel: DeviceSelector) -> Result<Receiver<AudioFrame>>;
}
