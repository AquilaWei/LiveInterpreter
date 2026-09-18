//! `cargo run -p li-vad --example gate -- <clip.wav>` — run the gate over a
//! 16 kHz mono wav and report what it decided and what it cost. Used to check
//! the budget (< 50 ms per window) and to eyeball threshold changes.

use std::time::Instant;

use anyhow::{Result, bail};
use li_types::AudioFrame;
use li_vad::{GateConfig, SileroVad, Vad, VadEvent};

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        bail!("usage: gate <clip.wav>");
    };
    let mut r = hound::WavReader::open(&path)?;
    if r.spec().sample_rate != 16_000 || r.spec().channels != 1 {
        bail!("need 16 kHz mono, got {:?}", r.spec());
    }
    let pcm: Vec<f32> = r
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let mut vad = SileroVad::new(GateConfig::default())?;
    let (mut speech, mut windows, mut runs) = (0usize, 0usize, 0usize);
    let mut cost = Vec::new();
    let mut open_at = None;
    let mut spoken = 0.0f64;

    for (i, chunk) in pcm.chunks(512).enumerate() {
        let t = i as f64 * 512.0 / 16_000.0;
        let frame = AudioFrame {
            pcm: chunk.to_vec(),
            sample_rate: 16_000,
            t_capture: Instant::now(),
        };
        let t0 = Instant::now();
        let events = vad.push(&frame)?;
        if chunk.len() == 512 {
            cost.push(t0.elapsed().as_secs_f64() * 1000.0);
            windows += 1;
            if vad.is_speech() {
                speech += 1;
            }
        }
        for e in events {
            match e {
                VadEvent::SpeechStart => {
                    runs += 1;
                    open_at = Some(t);
                }
                VadEvent::SpeechEnd => {
                    if let Some(s) = open_at.take() {
                        spoken += t - s;
                    }
                }
            }
        }
    }
    cost.sort_by(f64::total_cmp);
    let audio_s = pcm.len() as f64 / 16_000.0;
    let total: f64 = cost.iter().sum();
    println!("{path}: {audio_s:.1}s audio");
    println!(
        "  speech {:.1}% of windows, {runs} runs, {spoken:.1}s inside them",
        100.0 * speech as f64 / windows as f64
    );
    println!(
        "  per window: median {:.2} ms, p99 {:.2} ms, max {:.2} ms  (budget 32 ms)",
        cost[cost.len() / 2],
        cost[cost.len() * 99 / 100],
        cost[cost.len() - 1]
    );
    println!(
        "  RTF {:.4} ({:.0}x real time on one thread)",
        total / 1000.0 / audio_s,
        audio_s / (total / 1000.0)
    );
    Ok(())
}
