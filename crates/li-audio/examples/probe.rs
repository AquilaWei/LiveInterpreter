//! `cargo run -p li-audio --example probe [device]` — list capture devices and
//! sample one of them for two seconds, to confirm the platform backend actually
//! works here. With no argument it takes the system loopback (whatever the
//! machine is playing). Reports frame counts and levels only; no audio is kept.

use anyhow::Result;
use li_audio::{AudioSource, DesktopSource};
use li_types::DeviceSelector;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    println!("capture devices:");
    for d in DesktopSource::list_devices()? {
        println!(
            "  {} {}",
            if d.is_loopback {
                "[loopback]"
            } else {
                "[input]   "
            },
            d.name
        );
    }

    let sel = match std::env::args().nth(1).as_deref() {
        Some("mic") => DeviceSelector::Microphone,
        Some(name) => DeviceSelector::Device(name.to_owned()),
        None => DeviceSelector::SystemLoopback,
    };
    println!("\nsampling {sel:?} for 2s...");
    let mut src = DesktopSource::new();
    let mut rx = src.open(sel).await?;

    let (mut frames, mut samples, mut peak) = (0u32, 0usize, 0.0f32);
    // The clock starts on the first frame: a monitor of a suspended sink can
    // take a moment to spin up, and we want to measure the stream, not the open.
    // A monitor of a sink that is playing nothing may never produce one, so the
    // wait is bounded -- reporting zero frames is the useful answer there.
    let first = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .unwrap_or_else(|_| {
            println!("  (no frame within 5s -- is anything playing to this source?)");
            None
        });
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut frame = first;
    while let Some(f) = frame {
        frames += 1;
        samples += f.pcm.len();
        peak = peak.max(f.pcm.iter().fold(0.0f32, |m, s| m.max(s.abs())));
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(next) => frame = next,
            Err(_) => break,
        }
    }
    println!(
        "  {frames} frames, {samples} samples @16k ({:.2}s), peak {peak:.4}",
        samples as f32 / 16_000.0
    );
    Ok(())
}
