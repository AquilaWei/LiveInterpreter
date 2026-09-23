//! An audio file in, a finished transcript out.
//!
//! ## Why this is not [`crate::Engine::start_from_wav`]
//!
//! The live pipeline is built to drop things. The accurate lane has a queue
//! of two and a full queue means the fast lane's words are used instead,
//! and a translation draft that has been overtaken is thrown away. That is
//! right for subtitles, where the newest line is what matters, and wrong for a
//! file, where every sentence matters and nobody is waiting on any single one.
//!
//! So this path shares the parts and none of the policy. It is one loop on
//! one thread: cut the clip at its pauses, run every piece through the
//! accurate lane, translate every line, write the file. Nothing is dropped and
//! nothing is raced.
//!
//! ```text
//! decode -> Silero per 32 ms window -> segments() -> whisper -> NLLB -> .txt
//! ```

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use li_asr::{AsrEngine, DEFAULT_ENDPOINT_SILENCE_S, DEFAULT_MAX_UTTERANCE_S};
use li_audio::TARGET_RATE;
use li_transcript::render;
use li_types::{AsrEvent, AudioFrame, Lane, TranscriptLine};
use li_vad::{Gate, GateConfig, SileroVad, Vad, VadEvent, silero::WINDOW};
use serde::{Deserialize, Serialize};

use crate::config::EngineConfig;
use crate::models::{Kind, Models};

/// Which languages end up in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Output {
    /// The English transcript alone. Needs no translation model.
    En,
    /// The Chinese translation alone.
    Zh,
    /// Each English sentence with its Chinese under it.
    Both,
}

impl Output {
    /// The tail of the output filename, matching the live transcript's
    /// `_en.txt` and `_en-zh.txt`.
    fn suffix(self) -> &'static str {
        match self {
            Output::En => "en",
            Output::Zh => "zh",
            Output::Both => "en-zh",
        }
    }

    fn needs_translation(self) -> bool {
        self != Output::En
    }
}

impl std::str::FromStr for Output {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "en" => Ok(Output::En),
            "zh" => Ok(Output::Zh),
            "both" => Ok(Output::Both),
            other => bail!("output {other:?} is not one of en, zh, both"),
        }
    }
}

/// How far along a transcription is, in seconds of the input clip.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Progress {
    pub done_s: f64,
    pub total_s: f64,
}

/// Returned, wrapped in `anyhow`, when the caller set the cancel flag.
/// Check with `err.is::<Cancelled>()`: it is not a failure worth reporting.
#[derive(Debug, thiserror::Error)]
#[error("transcription cancelled")]
pub struct Cancelled;

/// Transcribe `input` and write the transcript into `[transcript] dir`.
///
/// Returns the path of the file written. It is named after the input --
/// `talk.mp3` becomes `talk_zh.txt` -- with `-2`, `-3`... added rather than
/// overwriting an earlier run.
///
/// Uses the accurate lane and the translation model from `cfg`; the fast lane
/// is not loaded. Fails if the accurate lane is `off`, if the models are not
/// downloaded, or if the file cannot be decoded. Nothing is left on disk
/// unless the whole transcript was written.
///
/// Blocks a thread for the whole run (minutes for a long file), so it runs on
/// the blocking pool. `cancel` is checked between sentences; `on_progress` is
/// called after each one.
pub async fn transcribe(
    cfg: EngineConfig,
    input: PathBuf,
    output: Output,
    cancel: Arc<AtomicBool>,
    on_progress: impl FnMut(Progress) + Send + 'static,
) -> Result<PathBuf> {
    let rt = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || run(&rt, &cfg, &input, output, &cancel, on_progress))
        .await
        .context("the transcription thread panicked")?
}

fn run(
    rt: &tokio::runtime::Handle,
    cfg: &EngineConfig,
    input: &Path,
    output: Output,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(Progress),
) -> Result<PathBuf> {
    let began = Instant::now();
    let lane = cfg
        .asr
        .accurate
        .as_ref()
        .filter(|l| l.backend != "off")
        .context("file transcription needs the accurate lane, and it is off in config.toml")?;
    // Where the models are, before anything slow: decoding an hour of audio
    // and only then reporting that nothing was downloaded wastes the hour.
    // Loading them waits until the audio has decoded, for the opposite
    // reason -- a file that is not audio should not cost a model load first.
    let models = Models::new();
    let spec = lane.to_spec(&models, Kind::Accurate, &cfg.asr.language)?;
    let nllb = if output.needs_translation() {
        if cfg.mt.backend == "off" {
            bail!("a Chinese transcript needs translation, and it is off in config.toml");
        }
        Some(cfg.mt.to_nllb(&models)?)
    } else {
        None
    };

    let clip = li_audio::decode::decode(input)?;
    let total_s = clip.duration_s();
    let probs = speech_probabilities(&clip.pcm)?;
    let pieces = segments(
        &probs,
        clip.pcm.len(),
        Duration::from_secs_f32(DEFAULT_ENDPOINT_SILENCE_S),
        Duration::from_secs_f32(DEFAULT_MAX_UTTERANCE_S),
    );
    tracing::info!(
        "{}: {total_s:.0} s of audio, {} pieces of speech",
        input.display(),
        pieces.len()
    );

    let mut asr = li_asr::build(&spec)?;
    tracing::info!("{}", asr.backend());
    let mt = nllb.map(|n| li_mt::LocalNllb::open(&n)).transpose()?;

    let mut text = String::new();
    for (n, piece) in pieces.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(Cancelled.into());
        }
        let Some(line) = recognise(rt, asr.as_mut(), &clip.pcm, piece, n as u64, &text)? else {
            continue;
        };
        let translation = match &mt {
            // Marks::Heard: whisper wrote this punctuation from the audio, the
            // same answer the live MT worker gives for an accurate-lane line.
            Some(mt) => mt.translate_blocking(&line.source, li_mt::chunk::Marks::Heard)?,
            None => String::new(),
        };
        text.push_str(&match output {
            Output::En => render::txt(&line),
            Output::Zh => render::txt_translation(&translation),
            Output::Both => render::bilingual(&line, &translation),
        });
        on_progress(Progress {
            done_s: samples_to_s(piece.end).min(total_s),
            total_s,
        });
    }
    // Cancelled during the last sentence: still cancelled, and still no file.
    if cancel.load(Ordering::Relaxed) {
        return Err(Cancelled.into());
    }
    on_progress(Progress {
        done_s: total_s,
        total_s,
    });

    let dir = li_transcript::expand_home(&cfg.transcript.dir);
    let path = write_new(&dir, &stem(input), output.suffix(), &text)?;
    tracing::info!(
        "wrote {} in {:.0} s",
        path.display(),
        began.elapsed().as_secs_f64()
    );
    Ok(path)
}

/// One piece of speech through whisper. `None` for a piece whisper heard as
/// nothing -- a cough, music the VAD let through.
fn recognise(
    rt: &tokio::runtime::Handle,
    asr: &mut dyn AsrEngine,
    pcm: &[f32],
    piece: &Range<usize>,
    line_id: u64,
    so_far: &str,
) -> Result<Option<TranscriptLine>> {
    // Carry the end of what has been said so far, as the live accurate lane
    // does (`li_stream::StreamConfig::prompt_chars`): it keeps names and
    // spellings consistent from one piece to the next.
    asr.set_prompt(prompt_tail(so_far, 200));
    let t_origin = Duration::from_secs_f64(samples_to_s(piece.start));
    let events = rt.block_on(asr.feed(&pcm[piece.clone()], t_origin))?;
    let Some((source, words)) = events.into_iter().find_map(|e| match e {
        AsrEvent::Partial { text, words, .. } => Some((text, words)),
        _ => None,
    }) else {
        return Ok(None);
    };
    let source = render::one_line(&source);
    if source.is_empty() {
        return Ok(None);
    }
    // Word times when whisper gave them, the piece's own span when it did not.
    let start_s = words
        .first()
        .map_or(samples_to_s(piece.start), |w| w.start.as_secs_f64());
    let end_s = words
        .last()
        .map_or(samples_to_s(piece.end), |w| w.end.as_secs_f64());
    Ok(Some(TranscriptLine {
        line_id,
        start_s,
        end_s,
        source,
        translation: None,
        lane: Lane::Accurate,
        reason: None,
    }))
}

/// Silero's speech probability for every [`WINDOW`] of the clip, in order.
fn speech_probabilities(pcm: &[f32]) -> Result<Vec<f32>> {
    let mut vad = SileroVad::new(GateConfig::default())?;
    let mut probs = Vec::with_capacity(pcm.len() / WINDOW + 1);
    for window in pcm.chunks(WINDOW) {
        let frame = AudioFrame {
            pcm: window.to_vec(),
            sample_rate: TARGET_RATE,
            t_capture: Instant::now(),
        };
        vad.push(&frame)?;
        probs.push(vad.probability());
    }
    Ok(probs)
}

/// Audio covered by one probability.
const WINDOW_DUR: Duration = Duration::from_millis(WINDOW as u64 * 1000 / TARGET_RATE as u64);

/// Audio kept on each side of a piece. Silero's probability rises a window or
/// two after a soft onset and falls before a trailing fricative; without this
/// whisper hears "ello" and "thi".
const PAD: usize = 6; // 192 ms

/// How far back from the length cap to look for somewhere quieter to cut.
const LOOKBACK: Duration = Duration::from_secs(3);

/// A stretch of speech, in windows. A side that is a forced cut gets no
/// [`PAD`]: the audio there belongs to the neighbouring piece, and handing it
/// to both makes whisper write the word on the boundary twice.
#[derive(Debug, Clone, PartialEq)]
struct Piece {
    windows: Range<usize>,
    cut_before: bool,
    cut_after: bool,
}

/// Where to cut the clip: sample ranges, one per piece of speech.
///
/// A piece ends after `pause` of silence, the same 0.6 s the live lanes end a
/// sentence on. A speaker who never pauses that long is cut before `max`, at
/// the quietest window in the [`LOOKBACK`] before it: live, that cut has to
/// land wherever the clock says, but a file can afford to look back for a
/// breath. Silence at either end of a piece is trimmed off and [`PAD`] put
/// back, so whisper is not handed long stretches of nothing to hallucinate
/// over.
///
/// `len` is the clip in samples. It is not `probs.len() * WINDOW`: the last
/// window is usually short, and a piece running to the end of the clip would
/// otherwise end past it.
///
/// A pure function of the probabilities, so the cutting can be tested without
/// the model.
fn segments(probs: &[f32], len: usize, pause: Duration, max: Duration) -> Vec<Range<usize>> {
    let cfg = GateConfig {
        min_silence: pause,
        ..GateConfig::default()
    };
    let max_windows = windows_in(max);
    let lookback = windows_in(LOOKBACK).min(max_windows - 1);
    let mut gate = Gate::new(cfg);
    let mut pieces: Vec<Piece> = Vec::new();
    // Where the open piece starts, and whether that start is a forced cut.
    let mut open: Option<(usize, bool)> = None;
    let mut last_voiced = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p >= cfg.silence_threshold {
            last_voiced = i;
        }
        match gate.push(p, WINDOW_DUR) {
            Some(VadEvent::SpeechStart) => open = Some((i, false)),
            Some(VadEvent::SpeechEnd) => {
                if let Some((s, cut_before)) = open.take() {
                    close(&mut pieces, s, cut_before, last_voiced);
                }
            }
            None => {
                if let Some((s, cut_before)) = open
                    && i + 1 - s >= max_windows
                {
                    let cut = quietest(probs, i + 1 - lookback..i + 1);
                    pieces.push(Piece {
                        windows: s..cut,
                        cut_before,
                        cut_after: true,
                    });
                    open = Some((cut, true));
                }
            }
        }
    }
    if let Some((s, cut_before)) = open {
        close(&mut pieces, s, cut_before, last_voiced);
    }
    pieces
        .iter()
        .map(|p| to_samples(p, probs.len(), len))
        .collect()
}

/// The earliest of the lowest-probability windows in `range`.
/// End the open piece, which starts at window `s`, after `last_voiced`.
///
/// A forced cut can land in the pause after a sentence, past the last voiced
/// window, when the cap falls while the gate is still counting that pause.
/// Nothing is left to transcribe after such a cut, so no piece is made; the
/// one before it ended in silence after all and gets its padding back.
fn close(pieces: &mut Vec<Piece>, s: usize, cut_before: bool, last_voiced: usize) {
    if s <= last_voiced {
        pieces.push(Piece {
            windows: s..last_voiced + 1,
            cut_before,
            cut_after: false,
        });
    } else if cut_before && let Some(before) = pieces.last_mut() {
        before.cut_after = false;
    }
}

fn quietest(probs: &[f32], range: Range<usize>) -> usize {
    let mut best = range.start;
    for i in range {
        if probs[i] < probs[best] {
            best = i;
        }
    }
    best
}

/// A piece's windows as samples, padded on the sides that are not cuts and
/// kept inside the clip.
fn to_samples(p: &Piece, windows: usize, len: usize) -> Range<usize> {
    let pad = |cut: bool| if cut { 0 } else { PAD };
    let a = p.windows.start.saturating_sub(pad(p.cut_before));
    let b = (p.windows.end + pad(p.cut_after)).min(windows);
    a * WINDOW..(b * WINDOW).min(len)
}

fn windows_in(d: Duration) -> usize {
    (d.as_millis() / WINDOW_DUR.as_millis()) as usize
}

fn samples_to_s(n: usize) -> f64 {
    n as f64 / TARGET_RATE as f64
}

/// The last `chars` characters of `text`, starting on a word.
fn prompt_tail(text: &str, chars: usize) -> &str {
    let count = text.chars().count();
    if count <= chars {
        return text.trim();
    }
    let (cut, _) = text.char_indices().nth(count - chars).expect("in range");
    let tail = &text[cut..];
    // Drop the word the cut landed in the middle of.
    match tail.find(char::is_whitespace) {
        Some(i) => tail[i..].trim(),
        None => tail.trim(),
    }
}

/// `talk.mp3` -> `talk`. A file with no usable name still gets one.
fn stem(input: &Path) -> String {
    input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "transcript".into())
}

/// Write `text` to `<dir>/<stem>_<suffix>.txt`, or `-2`, `-3`... after it if
/// that is taken. Written under a temporary name and renamed into place, so a
/// crash mid-write never leaves something that looks like a finished file.
fn write_new(dir: &Path, stem: &str, suffix: &str, text: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create transcript directory {}", dir.display()))?;
    let tmp = dir.join(format!(".{stem}_{suffix}.txt.part"));
    std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
    for n in 1..1000 {
        let name = if n == 1 {
            format!("{stem}_{suffix}.txt")
        } else {
            format!("{stem}_{suffix}-{n}.txt")
        };
        let path = dir.join(name);
        // `hard_link` fails if the target exists, which `rename` would not:
        // this is the check and the claim in one step.
        match std::fs::hard_link(&tmp, &path) {
            Ok(()) => {
                std::fs::remove_file(&tmp).ok();
                return Ok(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                std::fs::remove_file(&tmp).ok();
                return Err(e).with_context(|| format!("create {}", path.display()));
            }
        }
    }
    std::fs::remove_file(&tmp).ok();
    bail!(
        "{} already holds a thousand transcripts of {stem}",
        dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAUSE: Duration = Duration::from_millis(600);
    const MAX: Duration = Duration::from_secs(12);

    fn probs(runs: &[(f32, usize)]) -> Vec<f32> {
        runs.iter()
            .flat_map(|&(p, n)| std::iter::repeat_n(p, n))
            .collect()
    }

    #[test]
    fn two_sentences_with_a_long_pause_between_them_are_two_pieces() {
        // 50 windows of speech, 22 of silence (0.70 s), 30 of speech.
        let p = probs(&[(0.0, 10), (0.9, 50), (0.0, 22), (0.9, 30), (0.0, 10)]);

        let got = segments(&p, 122 * 512, PAUSE, MAX);

        // Windows 4..66 and 76..118, times 512 samples.
        assert_eq!(got, vec![2048..33792, 38912..60416]);
    }

    #[test]
    fn a_short_pause_inside_a_sentence_does_not_cut_it() {
        // 10 windows of silence is 0.32 s.
        let p = probs(&[(0.0, 10), (0.9, 50), (0.0, 10), (0.9, 30), (0.0, 10)]);

        let got = segments(&p, 110 * 512, PAUSE, MAX);

        // Windows 4..106.
        assert_eq!(got, vec![2048..54272]);
    }

    #[test]
    fn fifteen_seconds_of_even_speech_is_cut_three_seconds_before_the_cap() {
        // 469 windows is 15.0 s; the cap is 375, and the look-back 93.
        let p = probs(&[(0.9, 469)]);

        let got = segments(&p, 469 * 512, PAUSE, MAX);

        // Windows 0..282 and 282..469.
        assert_eq!(got, vec![0..144384, 144384..240128]);
    }

    #[test]
    fn a_long_sentence_is_cut_at_its_quietest_moment_before_the_cap() {
        // A breath at 9.6 s: quieter than speech, not quiet enough to be a pause.
        let p = probs(&[(0.9, 300), (0.4, 6), (0.9, 163)]);

        let got = segments(&p, 469 * 512, PAUSE, MAX);

        // Windows 0..300 and 300..469.
        assert_eq!(got, vec![0..153600, 153600..240128]);
    }

    #[test]
    fn the_two_sides_of_a_forced_cut_do_not_overlap() {
        // What this prevents: whisper writing "what effects" at the end of
        // one line and again at the start of the next.
        let p = probs(&[(0.0, 20), (0.9, 469), (0.0, 20)]);

        let got = segments(&p, 509 * 512, PAUSE, MAX);

        assert_eq!(got[0].end, got[1].start);
    }

    #[test]
    fn a_cap_reached_in_the_pause_after_a_sentence_leaves_one_piece() {
        // The sentence stops just short of 12 s, and the cap falls while the
        // gate is still counting the pause. The quietest window is late in
        // that pause, after the last voiced one: the piece after the cut
        // started past its own end, and slicing it panicked (a 24-minute
        // recording, 2026-09-23).
        let p = probs(&[(0.0, 20), (0.9, 358), (0.2, 10), (0.1, 40)]);

        let got = segments(&p, 428 * 512, PAUSE, MAX);

        // Windows 14..394: the cut at 388, the first 0.1, with the end padding
        // a pause earns.
        assert_eq!(got, vec![7168..201728]);
    }

    #[test]
    fn a_clip_of_silence_has_no_pieces() {
        let p = probs(&[(0.1, 300)]);

        assert_eq!(
            segments(&p, 300 * 512, PAUSE, MAX),
            Vec::<Range<usize>>::new()
        );
    }

    #[test]
    fn speech_running_to_the_end_of_the_clip_is_kept() {
        let p = probs(&[(0.0, 20), (0.9, 40)]);

        // Windows 14..60: the end padding stops at the clip.
        assert_eq!(segments(&p, 60 * 512, PAUSE, MAX), vec![7168..30720]);
    }

    #[test]
    fn a_piece_ending_in_a_short_last_window_stops_at_the_last_sample() {
        // 11.0 s of audio is 343 full windows and 384 samples over.
        let p = probs(&[(0.0, 20), (0.9, 324)]);

        assert_eq!(segments(&p, 176_000, PAUSE, MAX), vec![7168..176_000]);
    }

    /// What this prevents: an hour of audio decoded, and only then the news
    /// that the models were never downloaded.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_missing_model_is_reported_before_the_audio_is_read() {
        let mut cfg = EngineConfig::default();
        cfg.asr.accurate.as_mut().unwrap().model = "/nonexistent/ggml-small.en.bin".into();

        let err = transcribe(
            cfg,
            "/nonexistent/talk.mp3".into(),
            Output::En,
            Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .await
        .unwrap_err();

        assert!(format!("{err:#}").contains("model not found"), "{err:#}");
    }

    #[test]
    fn output_names_are_the_ones_the_cli_takes() {
        assert_eq!("en".parse::<Output>().unwrap(), Output::En);
        assert_eq!("zh".parse::<Output>().unwrap(), Output::Zh);
        assert_eq!("both".parse::<Output>().unwrap(), Output::Both);
    }

    #[test]
    fn an_unknown_output_name_is_rejected_with_the_real_ones() {
        let err = "cn".parse::<Output>().unwrap_err().to_string();

        assert!(err.contains("en, zh, both"), "{err}");
    }

    #[test]
    fn a_short_prompt_is_kept_whole() {
        assert_eq!(prompt_tail("Ask not. ", 200), "Ask not.");
    }

    #[test]
    fn a_long_prompt_is_cut_on_a_word_boundary() {
        assert_eq!(prompt_tail("what your country can do", 10), "can do");
    }

    #[test]
    fn the_stem_is_the_input_filename_without_its_extension() {
        assert_eq!(stem(Path::new("/music/Episode 12.mp3")), "Episode 12");
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("li_batch_test").join(name);
        std::fs::remove_dir_all(&dir).ok();
        dir
    }

    #[test]
    fn the_first_transcript_of_a_file_takes_the_plain_name() {
        let dir = scratch("plain");

        let path = write_new(&dir, "talk", "zh", "你好\n").unwrap();

        assert_eq!(path, dir.join("talk_zh.txt"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "你好\n");
    }

    #[test]
    fn a_second_transcript_does_not_overwrite_the_first() {
        let dir = scratch("second");
        write_new(&dir, "talk", "zh", "first\n").unwrap();

        let path = write_new(&dir, "talk", "zh", "second\n").unwrap();

        assert_eq!(path, dir.join("talk_zh-2.txt"));
        assert_eq!(
            std::fs::read_to_string(dir.join("talk_zh.txt")).unwrap(),
            "first\n"
        );
    }

    #[test]
    fn no_temporary_file_is_left_behind() {
        let dir = scratch("tmp");

        write_new(&dir, "talk", "en", "hi\n").unwrap();

        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["talk_en.txt".to_string()]);
    }
}
