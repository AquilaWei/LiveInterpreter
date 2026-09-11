//! A WAV file pretending to be a capture device.
//!
//! This is what the evaluation harness records against, so it has to behave
//! like hardware in the one way that matters: it delivers frames on a fixed
//! schedule and does **not** slow down when the consumer falls behind. A file
//! source that waits for a slow ASR pass would hide exactly the failure Phase 0
//! hit -- the rolling buffer growing without bound because the pipeline could
//! not keep up -- and every latency number measured against it would be a
//! fiction.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use li_types::AudioFrame;
use tokio::sync::mpsc;

use crate::{FRAME_MS, TARGET_RATE, agc::Agc, resample::Resampler};

pub struct FileSource {
    path: PathBuf,
    agc: bool,
    /// Deliver as fast as possible instead of at playback speed. For batch
    /// scoring, where wall-clock latency is not being measured.
    realtime: bool,
}

impl FileSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            agc: true,
            realtime: true,
        }
    }

    pub fn with_agc(mut self, on: bool) -> Self {
        self.agc = on;
        self
    }

    pub fn with_realtime(mut self, on: bool) -> Self {
        self.realtime = on;
        self
    }

    pub fn open(self) -> Result<mpsc::Receiver<AudioFrame>> {
        let reader = hound::WavReader::open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let spec = reader.spec();
        let (tx, rx) = mpsc::channel(64);

        std::thread::spawn(move || {
            if let Err(e) = pump(reader, spec, self.agc, self.realtime, tx) {
                tracing::error!("file source stopped: {e:#}");
            }
        });
        Ok(rx)
    }
}

fn pump(
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
    spec: hound::WavSpec,
    agc_on: bool,
    realtime: bool,
    tx: mpsc::Sender<AudioFrame>,
) -> Result<()> {
    let mut resampler = Resampler::new(spec.sample_rate, TARGET_RATE);
    let mut agc = Agc::default();
    let frame_len = (TARGET_RATE as usize * FRAME_MS as usize) / 1000;
    // Read at the source rate in chunks that yield about one frame each.
    let chunk = (frame_len * spec.sample_rate as usize / TARGET_RATE as usize).max(1)
        * spec.channels as usize;

    let samples: Box<dyn Iterator<Item = Result<f32, hound::Error>>> = match spec.sample_format {
        hound::SampleFormat::Float => Box::new(reader.into_samples::<f32>()),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            Box::new(
                reader
                    .into_samples::<i32>()
                    .map(move |s| s.map(|v| v as f32 * scale)),
            )
        }
    };

    let start = Instant::now();
    let mut emitted: u32 = 0;
    let mut buf = Vec::with_capacity(chunk);
    let mut out = Vec::new();

    for s in samples {
        buf.push(s?);
        if buf.len() < chunk {
            continue;
        }
        out.extend(resampler.process(&buf, spec.channels));
        buf.clear();

        while out.len() >= frame_len {
            let mut pcm: Vec<f32> = out.drain(..frame_len).collect();
            if agc_on {
                agc.process(&mut pcm);
            }
            emitted += 1;
            if realtime {
                let due = Duration::from_millis((emitted as u64) * FRAME_MS as u64);
                // If the consumer fell behind, the backlog goes out immediately:
                // real hardware cannot pause the world either.
                if let Some(wait) = due.checked_sub(start.elapsed()) {
                    std::thread::sleep(wait);
                }
            }
            if tx
                .blocking_send(AudioFrame {
                    pcm,
                    sample_rate: TARGET_RATE,
                    t_capture: Instant::now(),
                })
                .is_err()
            {
                return Ok(()); // consumer went away
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(path: &Path, rate: u32, channels: u16, secs: f32) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        let n = (rate as f32 * secs) as usize;
        for i in 0..n {
            let v = (0.2
                * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin()
                * i16::MAX as f32) as i16;
            for _ in 0..channels {
                w.write_sample(v).unwrap();
            }
        }
        w.finalize().unwrap();
    }

    #[tokio::test]
    async fn a_48k_stereo_file_arrives_as_16k_mono_frames() {
        let dir = std::env::temp_dir().join("li_audio_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");
        write_wav(&path, 48_000, 2, 1.0);

        let mut rx = FileSource::new(&path).with_realtime(false).open().unwrap();
        let mut frames = 0;
        let mut samples = 0;
        while let Some(f) = rx.recv().await {
            assert_eq!(f.sample_rate, TARGET_RATE);
            assert_eq!(f.pcm.len(), 512); // 32 ms at 16 kHz
            frames += 1;
            samples += f.pcm.len();
        }
        assert!(frames > 25, "only {frames} frames");
        assert!((samples as i64 - 16_000).abs() < 600, "{samples} samples");
        std::fs::remove_file(&path).ok();
    }
}
