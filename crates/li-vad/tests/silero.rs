//! The ONNX half, against real audio. `gate.rs` covers the decision logic on
//! synthetic probabilities; what is left to check here is that the model is
//! actually being driven correctly, which is not something a shape assertion
//! can tell you -- fed wrongly, Silero returns confident silence rather than an
//! error.

use std::{path::PathBuf, time::Duration, time::Instant};

use li_types::AudioFrame;
use li_vad::{GateConfig, SileroVad, Vad, VadEvent};

fn testdata(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn read_wav(name: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(testdata(name)).expect("test clip");
    assert_eq!(r.spec().sample_rate, 16_000);
    assert_eq!(r.spec().channels, 1);
    r.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn frames(pcm: &[f32], len: usize) -> Vec<AudioFrame> {
    pcm.chunks(len)
        .map(|c| AudioFrame {
            pcm: c.to_vec(),
            sample_rate: 16_000,
            t_capture: Instant::now(),
        })
        .collect()
}

/// Windows scored as speech, and the events emitted along the way.
fn run(vad: &mut SileroVad, pcm: &[f32], frame_len: usize) -> (f64, Vec<VadEvent>) {
    let mut speech = 0usize;
    let mut total = 0usize;
    let mut events = Vec::new();
    for f in frames(pcm, frame_len) {
        let n = f.pcm.len();
        events.extend(vad.push(&f).expect("vad"));
        // Count per window, not per frame, by sampling the state after each
        // frame: frame and window are both 512 samples in this project.
        if n == frame_len {
            total += 1;
            if vad.is_speech() {
                speech += 1;
            }
        }
    }
    (speech as f64 / total as f64, events)
}

#[test]
fn clean_speech_is_recognised_as_speech() {
    // The regression this guards: Silero needs 64 samples of history prepended
    // to each 512-sample window. Without them it does not fail -- it returns
    // ~0.001 for every window of a person clearly talking. This clip is 85%
    // speech by the reference implementation; a wiring mistake reads as 0%.
    let mut vad = SileroVad::new(GateConfig::default()).unwrap();
    let (fraction, events) = run(&mut vad, &read_wav("read_clean.wav"), 512);
    assert!(
        fraction > 0.7,
        "speech fraction {fraction:.3} -- is the context window wired up?"
    );
    assert!(
        fraction < 0.98,
        "speech fraction {fraction:.3} -- the gate never closes"
    );
    // The clip is separate utterances joined by 0.7s gaps, so the gate should
    // open and close several times rather than latching once.
    let starts = events
        .iter()
        .filter(|e| **e == VadEvent::SpeechStart)
        .count();
    assert!(
        starts >= 5,
        "only {starts} speech runs in 82s of read speech"
    );
}

#[test]
fn a_far_field_meeting_is_recognised_as_speech() {
    let mut vad = SileroVad::new(GateConfig::default()).unwrap();
    let (fraction, _) = run(&mut vad, &read_wav("ami_meeting.wav"), 512);
    assert!(
        fraction > 0.5,
        "speech fraction {fraction:.3} on a meeting recording"
    );
}

#[test]
fn silence_never_opens_the_gate() {
    let mut vad = SileroVad::new(GateConfig::default()).unwrap();
    let (fraction, events) = run(&mut vad, &vec![0.0; 16_000 * 3], 512);
    assert_eq!(fraction, 0.0);
    assert_eq!(events, vec![]);
    assert!(!vad.is_speech());
    assert!(
        vad.silence() >= Duration::from_secs(2),
        "silence {:?}",
        vad.silence()
    );
}

#[test]
fn the_result_does_not_depend_on_how_the_audio_is_chopped_up() {
    // `li-audio` promises 512-sample frames, but a file source or a different
    // backend need not, and the window buffering has to absorb that.
    let pcm = read_wav("read_clean.wav");
    let mut a = SileroVad::new(GateConfig::default()).unwrap();
    let mut b = SileroVad::new(GateConfig::default()).unwrap();
    let (fa, ea) = run(&mut a, &pcm, 512);
    let (_, eb) = run(&mut b, &pcm, 1000);
    assert_eq!(ea, eb, "event sequence changed with the frame size");
    assert!(fa > 0.7);
}

#[test]
fn the_wrong_sample_rate_is_an_error_not_a_wrong_answer() {
    let mut vad = SileroVad::new(GateConfig::default()).unwrap();
    let err = vad
        .push(&AudioFrame {
            pcm: vec![0.0; 512],
            sample_rate: 48_000,
            t_capture: Instant::now(),
        })
        .unwrap_err();
    assert!(err.to_string().contains("16000"), "{err}");
}
