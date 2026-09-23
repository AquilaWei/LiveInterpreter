//! Both lanes against real audio.
//!
//! These need model files, which are downloaded rather than committed (they are
//! 179 MiB and 190 MiB). When they are absent the test prints a SKIP line and
//! passes, so CI stays green on a machine with no models — a silent skip would
//! be worse than no test, so the line is loud and says what is missing.
//!
//!     LI_SHERPA_MODEL=... LI_WHISPER_MODEL=... cargo test -p li-asr -- --nocapture

use std::{path::PathBuf, time::Duration};

use li_asr::{AsrEngine, LaneSpec};
use li_types::AsrEvent;

const FRAME: usize = 512;

fn model(env: &str, default: &str, what: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let p = std::env::var(env)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(&home).join(default));
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP: no {what} model at {} (set ${env})", p.display());
        None
    }
}

fn fast_model() -> Option<PathBuf> {
    model(
        "LI_SHERPA_MODEL",
        ".cache/liveinterpreter/models/sherpa-onnx-streaming-zipformer-en-2023-06-21",
        "fast-lane",
    )
}

fn accurate_model() -> Option<PathBuf> {
    model(
        "LI_WHISPER_MODEL",
        ".cache/liveinterpreter/models/ggml/ggml-small.en-q5_1.bin",
        "accurate-lane",
    )
}

fn clip(name: &str) -> Option<Vec<f32>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    let mut r = hound::WavReader::open(&path).ok()?;
    assert_eq!(r.spec().sample_rate, 16_000, "{name} must be 16 kHz");
    Some(
        r.samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / f32::from(i16::MAX))
            .collect(),
    )
}

fn at(samples: usize) -> Duration {
    Duration::from_secs_f64(samples as f64 / 16_000.0)
}

/// Drive a lane over a clip in fixed frames, returning the final sentences.
async fn run(
    engine: &mut dyn AsrEngine,
    pcm: &[f32],
    frame: usize,
) -> Vec<(String, Duration, Duration)> {
    let mut out = Vec::new();
    let mut fed = 0usize;
    for chunk in pcm.chunks(frame) {
        for ev in engine.feed(chunk, at(fed)).await.unwrap() {
            if let AsrEvent::Final {
                words,
                t_start,
                t_end,
                ..
            } = ev
            {
                out.push((li_asr::words::text_of(&words), t_start, t_end));
            }
        }
        fed += chunk.len();
    }
    if let Some(AsrEvent::Final {
        words,
        t_start,
        t_end,
        ..
    }) = engine.finalize().await.unwrap()
    {
        out.push((li_asr::words::text_of(&words), t_start, t_end));
    }
    out
}

#[tokio::test]
async fn the_fast_lane_transcribes_and_breaks_sentences() {
    let (Some(m), Some(pcm)) = (fast_model(), clip("read_clean.wav")) else {
        return;
    };
    let mut e = li_asr::build(&LaneSpec::new("sherpa", m)).unwrap();
    assert!(e.capabilities().native_streaming);
    // The fast lane's model has no punctuation or casing; anything downstream
    // that splits on "." would see one 82-second sentence.
    assert!(!e.capabilities().punctuated);

    let lines = run(e.as_mut(), &pcm, FRAME).await;
    assert!(
        lines.len() >= 5,
        "82 s of read speech should break into several lines, got {}",
        lines.len()
    );
    let all = lines
        .iter()
        .map(|(t, _, _)| t.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    for word in ["MEETING", "BROTHER", "PRISON"] {
        assert!(all.contains(word), "expected {word:?} in: {all}");
    }
    // The word-boundary marker must never survive into the text: it did, as a
    // missing *space*, until the Rust output was diffed against the PoC's.
    assert!(!all.contains('\u{2581}'));
    assert!(!all.contains("  "));
}

#[tokio::test]
async fn sentence_spans_are_real_and_ordered() {
    let (Some(m), Some(pcm)) = (fast_model(), clip("read_clean.wav")) else {
        return;
    };
    let mut e = li_asr::build(&LaneSpec::new("sherpa", m)).unwrap();
    let lines = run(e.as_mut(), &pcm, FRAME).await;

    let mut last_end = Duration::ZERO;
    for (text, start, end) in &lines {
        // The prototype's bug: t_start == t_end, and every SRT block was zero-length.
        assert!(end > start, "zero-length span for {text:?}");
        assert!(*start >= last_end, "sentences out of order at {text:?}");
        assert!(
            *end <= at(pcm.len()) + Duration::from_secs(1),
            "{text:?} ends after the clip does"
        );
        last_end = *end;
    }
}

#[tokio::test]
async fn the_fast_lane_does_not_care_how_the_audio_is_chopped_up() {
    let (Some(m), Some(pcm)) = (fast_model(), clip("read_clean.wav")) else {
        return;
    };
    // Frames come from a device, not from us: `li-audio` delivers 32 ms, a wav
    // file delivers whatever is left at the end, and a stall delivers a burst.
    // A recogniser whose output depends on that is not usable here.
    async fn text(model: &std::path::Path, pcm: &[f32], frame: usize) -> String {
        let mut e = li_asr::build(&LaneSpec::new("sherpa", model)).unwrap();
        run(e.as_mut(), pcm, frame)
            .await
            .into_iter()
            .map(|(t, _, _)| t)
            .collect::<Vec<_>>()
            .join(" ")
    }
    assert_eq!(
        text(&m, &pcm, 512).await,
        text(&m, &pcm, 1600).await,
        "the transcript changed when the frame size did"
    );
}

#[tokio::test]
async fn the_timeline_follows_t_origin_not_the_samples_fed() {
    let (Some(m), Some(pcm)) = (fast_model(), clip("read_clean.wav")) else {
        return;
    };
    // `li-audio` drops frames when the pipeline is behind, so "samples fed" and
    // "seconds of audio that happened" diverge. Feed only the first half of
    // every second but keep telling the engine the true time, and the reported
    // spans must follow the true time.
    let mut e = li_asr::build(&LaneSpec::new("sherpa", m)).unwrap();
    let mut out = Vec::new();
    let mut fed = 0usize;
    for (i, chunk) in pcm.chunks(FRAME).enumerate() {
        if i % 2 == 1 {
            fed += chunk.len(); // pretend this frame was dropped
            continue;
        }
        for ev in e.feed(chunk, at(fed)).await.unwrap() {
            if let AsrEvent::Final { t_start, .. } = ev {
                out.push(t_start);
            }
        }
        fed += chunk.len();
    }
    let last = out.last().copied().unwrap_or_default();
    // Without the drift correction the last sentence would land at about half
    // the clip length, because sherpa only ever saw half the audio.
    assert!(
        last > at(pcm.len()) / 2,
        "last sentence at {last:?} of a {:?} clip -- the timeline is following \
         the samples fed, not the clock",
        at(pcm.len())
    );
}

#[tokio::test]
async fn the_accurate_lane_punctuates() {
    let (Some(m), Some(pcm)) = (accurate_model(), clip("jfk.wav")) else {
        return;
    };
    let mut e = li_asr::build(&LaneSpec::new("whispercpp", m)).unwrap();
    assert!(!e.capabilities().native_streaming);
    assert!(e.capabilities().punctuated);

    let evs = e.feed(&pcm, Duration::ZERO).await.unwrap();
    let AsrEvent::Partial { text, words, .. } = evs.first().expect("a hypothesis") else {
        panic!("the accurate lane emits hypotheses, never finals, until finalize()");
    };
    assert!(text.contains("fellow Americans"), "{text}");
    // LocalAgreement-2 compares words, so a hypothesis without them is useless
    // to `li-stream` no matter how good the string is.
    assert!(!words.is_empty());
    assert!(words.windows(2).all(|w| w[1].start >= w[0].start));

    let AsrEvent::Final { t_end, .. } = e.finalize().await.unwrap().expect("a final") else {
        panic!("finalize must produce a Final");
    };
    assert!(t_end <= at(pcm.len()) + Duration::from_secs(1));
}

#[test]
fn a_missing_model_names_the_path_and_the_setting() {
    let err = match li_asr::build(&LaneSpec::new("sherpa", "/nonexistent/model-dir")) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a model directory that does not exist must not load"),
    };
    assert!(err.contains("/nonexistent/model-dir"), "{err}");
    assert!(err.contains("asr.fast"), "{err}");
}

#[test]
fn an_unknown_backend_says_what_the_valid_ones_are() {
    let err = match li_asr::build(&LaneSpec::new("deepgram", "/tmp")) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("an unknown backend name must not load"),
    };
    assert!(err.contains("sherpa"), "{err}");
}

/// The punctuation model against the shape of text it will actually be given:
/// upper case, no marks, straight off the fast lane.
///
/// Skips like the two above when the model is absent. What it pins is the pair
/// of assumptions `li_asr::punct` is built on and neither the C API nor the
/// model card states — that lower-casing the input is required to get any
/// casing back, and that the word count survives.
#[test]
fn punctuation_restores_marks_and_casing_without_moving_the_words() {
    let Some(dir) = model(
        "LI_PUNCT_MODEL",
        ".cache/liveinterpreter/models/sherpa-onnx-online-punct-en-2024-08-06",
        "punctuation",
    ) else {
        return;
    };
    let mut p = li_asr::punct::OnlinePunct::open(&dir, 1).expect("opening the punctuation model");

    let shouted = "NOTHING CAN SURPRISE ME NOW I'M PREPARED FOR ANYTHING";
    let out = p.restore(shouted).expect("restoring");
    eprintln!("punct: {shouted}\n    -> {out}");

    assert!(
        out.contains(['.', '?', '!']),
        "no sentence-final mark came back: {out}"
    );
    assert!(
        out.chars().any(|c| c.is_lowercase()),
        "everything came back upper case, so the input was not lowered: {out}"
    );
    assert_eq!(
        out.split_whitespace().count(),
        shouted.split_whitespace().count(),
        "the word count moved, which `restore_words` refuses to ship: {out}"
    );

    // And the guard itself: `restore_words` must hand back timed words whose
    // times are untouched.
    let words: Vec<li_types::Word> = shouted
        .split_whitespace()
        .enumerate()
        .map(|(i, t)| li_types::Word {
            text: t.to_owned(),
            start: Duration::from_millis(i as u64 * 200),
            end: Duration::from_millis(i as u64 * 200 + 200),
        })
        .collect();
    let back = p.restore_words(&words).expect("restoring words");
    assert_eq!(back.len(), words.len());
    assert_eq!(p.mismatches(), 0, "the word-count guard tripped");
    for (a, b) in words.iter().zip(&back) {
        assert_eq!((a.start, a.end), (b.start, b.end), "times must not move");
    }
    eprintln!(
        "punct words: {}",
        back.iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    );
}

/// What this prevents: a segfault. Two whisper models loading onto Vulkan at
/// the same moment crashed inside `ggml_backend_alloc_ctx_tensors_from_buft`,
/// which called a null function pointer -- the Vulkan backend's
/// initialisation is not thread-safe. Found by the file transcriber's tests,
/// which load one whisper each and run in parallel. On a CPU build this
/// passes either way; the crash needs `--features gpu-vulkan`.
#[test]
fn four_accurate_lanes_can_load_at_the_same_time() {
    let Some(m) = accurate_model() else { return };

    let loads: Vec<_> = (0..4)
        .map(|_| {
            let m = m.clone();
            std::thread::spawn(move || li_asr::build(&LaneSpec::new("whispercpp", m)).is_ok())
        })
        .collect();

    for l in loads {
        assert!(l.join().unwrap(), "a load failed");
    }
}
