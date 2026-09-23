//! File transcription end to end, on real models.
//!
//! Like `restart.rs`, it needs the downloaded models and `testdata/jfk.wav`,
//! and prints a SKIP line without them, so CI -- which has neither -- stays
//! green. The cutting, naming and rendering are unit tests in `batch.rs`; what
//! only this can show is that whisper and NLLB, driven the batch way, produce
//! the words.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use li_core::batch::{self, Cancelled, Output};
use li_core::{config::EngineConfig, download, models::Models};

const JFK: &str = "../../testdata/jfk.wav";

/// The default config writing into a fresh directory, or `None` when this
/// machine cannot run the test.
fn setup(name: &str) -> Option<EngineConfig> {
    let mut cfg = EngineConfig::default();
    let dir = std::env::temp_dir().join("li_batch_it").join(name);
    std::fs::remove_dir_all(&dir).ok();
    cfg.transcript.dir = dir;
    if !Path::new(JFK).exists() {
        eprintln!("SKIP: {JFK} is not here");
        return None;
    }
    let missing = download::plan(&Models::new(), &cfg).unwrap();
    if !missing.is_empty() {
        eprintln!(
            "SKIP: models not downloaded: {} (set $LI_MODEL_DIR)",
            missing.models().join(", ")
        );
        return None;
    }
    Some(cfg)
}

async fn run(cfg: EngineConfig, output: Output, cancel: bool) -> anyhow::Result<PathBuf> {
    batch::transcribe(
        cfg,
        JFK.into(),
        output,
        Arc::new(AtomicBool::new(cancel)),
        |_| {},
    )
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn an_english_transcript_has_the_speech_in_it() {
    let Some(cfg) = setup("en") else { return };

    let path = run(cfg, Output::En, false).await.unwrap();

    let text = std::fs::read_to_string(&path).unwrap().to_lowercase();
    assert!(path.ends_with("jfk_en.txt"), "{}", path.display());
    // Not "ask not what your country": Kennedy pauses 1.5 s after "ask not",
    // which is a sentence break by the 0.6 s rule.
    assert!(text.contains("what you can do for your country"), "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_chinese_transcript_is_in_chinese() {
    let Some(cfg) = setup("zh") else { return };

    let path = run(cfg, Output::Zh, false).await.unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("國家"), "{text}");
    assert!(!text.to_lowercase().contains("country"), "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bilingual_transcript_has_both_languages() {
    let Some(cfg) = setup("both") else { return };

    let path = run(cfg, Output::Both, false).await.unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    assert!(path.ends_with("jfk_en-zh.txt"), "{}", path.display());
    assert!(text.to_lowercase().contains("country"), "{text}");
    assert!(text.contains("國家"), "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_transcription_writes_no_file() {
    let Some(cfg) = setup("cancel") else { return };
    let dir = cfg.transcript.dir.clone();

    let err = run(cfg, Output::En, true).await.unwrap_err();

    assert!(err.is::<Cancelled>(), "{err:#}");
    assert!(!dir.exists(), "{} was created", dir.display());
}
