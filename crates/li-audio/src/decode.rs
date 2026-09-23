//! Any common audio file -> 16 kHz mono, all at once.
//!
//! For file transcription, not for playback: [`crate::file::FileSource`] is the
//! one that pretends to be a capture device and paces itself. This reads as
//! fast as the decoder goes and hands back the whole clip, because the batch
//! transcriber needs to see where the pauses are before it cuts anything.
//!
//! **Memory.** The clip is held in full: 16 000 f32 a second is about 230 MB
//! for an hour of audio. Accepted for now -- the models already take several
//! times that -- but it is the first thing to change if multi-hour files
//! become common.
//!
//! The level normalisation and the rate change are the same [`Agc`] and
//! [`Resampler`] the live path uses, so a file and a meeting reach the
//! recogniser looking alike.

use std::path::Path;

use anyhow::{Context, Result, bail};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::{FRAME_MS, TARGET_RATE, agc::Agc, resample::Resampler};

/// A decoded clip, ready for the models.
#[derive(Debug, Clone)]
pub struct Decoded {
    /// 16 kHz mono, level-normalised.
    pub pcm: Vec<f32>,
}

impl Decoded {
    pub fn duration_s(&self) -> f64 {
        self.pcm.len() as f64 / TARGET_RATE as f64
    }
}

/// Decode `path` (mp3, m4a/AAC, flac, ogg/vorbis, wav) to 16 kHz mono.
///
/// Fails when the file cannot be opened, is not a container this build knows,
/// has no audio track, or holds no audio at all. A packet that fails to decode
/// in the middle of a file is skipped rather than fatal: a damaged frame in a
/// long podcast should cost a few milliseconds, not the whole transcript.
pub fn decode(path: &Path) -> Result<Decoded> {
    let name = path.display();
    let file = std::fs::File::open(path).with_context(|| format!("opening {name}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .with_context(|| format!("{name} is not an audio format this build can read"))?;
    let track = format
        .default_track(TrackType::Audio)
        .with_context(|| format!("{name} has no audio track"))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .with_context(|| format!("{name}: the audio track has no codec parameters"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .with_context(|| format!("{name}: no decoder for this audio codec"))?;

    // Made on the first packet, because that is where the real rate and
    // channel count are known: container headers are allowed to omit them.
    let mut stream: Option<(u32, u16, Resampler)> = None;
    let mut interleaved: Vec<f32> = Vec::new();
    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(e).with_context(|| format!("reading {name}")),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(SymError::DecodeError(e)) => {
                tracing::warn!("{name}: skipped a damaged packet: {e}");
                continue;
            }
            Err(e) => return Err(e).with_context(|| format!("decoding {name}")),
        };
        let rate = buf.spec().rate();
        let channels = buf.spec().channels().count() as u16;
        let (want_rate, want_channels, resampler) =
            stream.get_or_insert_with(|| (rate, channels, Resampler::new(rate, TARGET_RATE)));
        if (rate, channels) != (*want_rate, *want_channels) {
            bail!("{name} changes its sample rate or channel count part way through");
        }
        buf.copy_to_vec_interleaved(&mut interleaved);
        mono.extend(resampler.process(&interleaved, channels));
    }
    if mono.is_empty() {
        bail!("{name} contains no audio");
    }

    normalise(&mut mono);
    Ok(Decoded { pcm: mono })
}

/// Run the live path's AGC over the clip in capture-sized frames, so its
/// time constants mean what they mean on a device.
fn normalise(pcm: &mut [f32]) {
    let frame_len = (TARGET_RATE * FRAME_MS / 1000) as usize;
    let mut agc = Agc::default();
    for frame in pcm.chunks_mut(frame_len) {
        agc.process(frame);
    }
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

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("li_audio_decode_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn two_seconds_of_48k_stereo_decode_to_32000_mono_samples() {
        let path = scratch("stereo48.wav");
        write_wav(&path, 48_000, 2, 2.0);

        let clip = decode(&path).unwrap();

        // The resampler's filter delay keeps a few samples back; within one
        // 32 ms frame is exact enough for every timestamp downstream.
        assert!(
            (clip.pcm.len() as i64 - 32_000).abs() < 512,
            "{}",
            clip.pcm.len()
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_16k_mono_file_keeps_its_length_exactly() {
        let path = scratch("mono16.wav");
        write_wav(&path, 16_000, 1, 1.5);

        let clip = decode(&path).unwrap();

        assert_eq!(clip.pcm.len(), 24_000);
        assert_eq!(clip.duration_s(), 1.5);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_file_is_an_error_naming_the_file() {
        let err = decode(Path::new("/nonexistent/talk.mp3")).unwrap_err();

        assert!(format!("{err:#}").contains("talk.mp3"), "{err:#}");
    }

    #[test]
    fn a_text_file_is_rejected_as_not_audio() {
        let path = scratch("notes.mp3");
        std::fs::write(&path, "this is not audio at all, only words\n".repeat(50)).unwrap();

        let err = decode(&path).unwrap_err();

        assert!(format!("{err:#}").contains("notes.mp3"), "{err:#}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_empty_wav_is_rejected_as_containing_no_audio() {
        let path = scratch("empty.wav");
        write_wav(&path, 16_000, 1, 0.0);

        let err = decode(&path).unwrap_err();

        assert!(format!("{err:#}").contains("no audio"), "{err:#}");
        std::fs::remove_file(&path).ok();
    }
}
