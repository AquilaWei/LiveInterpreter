//! Starting the engine again after stopping it (PLAN §17 task 1.23).
//!
//! This is what changing the audio source in the settings window now does:
//! `stop` and then `start`, because a running pipeline cannot be aimed at a
//! different capture device. The bet is that a second `start` works -- that
//! `load` does not still think a session is open, that the sink opens a fresh
//! transcript rather than reusing the closed one, and that nothing from the
//! first set of tasks is still holding the models.
//!
//! It costs about 25 s, and the reason is the clip: `start_from_wav` plays at
//! playback speed and `stop` waits for the pipeline to drain, so the test pays
//! for `jfk.wav` twice. Worth it -- the alternative is finding out from someone
//! whose meeting stopped transcribing when they switched microphones.
//!
//! Like `li-asr/tests/lanes.rs`, it needs the downloaded models and prints a
//! SKIP line without them, so CI -- which has none -- stays green. The check
//! is `download::plan`, the same one the app asks at startup.

use li_core::{Engine, config::EngineConfig, download, models::Models};

#[tokio::test(flavor = "multi_thread")]
async fn the_engine_can_be_started_again_after_being_stopped() {
    let dir = std::env::temp_dir().join("li_restart_test");
    std::fs::create_dir_all(&dir).unwrap();

    // The fast lane alone: the restart is what is under test, and whisper and
    // NLLB would only make it slower to find that out.
    let mut cfg = EngineConfig::default();
    cfg.asr.accurate = None;
    cfg.mt.backend = "off".into();
    cfg.transcript.enabled = true;
    cfg.transcript.dir = dir;
    let missing = download::plan(&Models::new(), &cfg).unwrap();
    if !missing.is_empty() {
        eprintln!(
            "SKIP: models not downloaded: {} (set $LI_MODEL_DIR)",
            missing.models().join(", ")
        );
        return;
    }
    let wav = std::path::Path::new("../../testdata/jfk.wav");

    let mut engine = Engine::new(cfg).unwrap();
    engine.start_from_wav(wav).await.expect("first start");
    let first = engine.transcripts().to_vec();
    engine.stop().await.expect("stop");

    engine.start_from_wav(wav).await.expect("start after stop");
    let second = engine.transcripts().to_vec();
    engine.stop().await.expect("stop again");

    assert!(!first.is_empty(), "the first session wrote no transcript");
    assert_ne!(
        first, second,
        "the second session reused the first one's transcript files"
    );
}
