//! What happens when NLLB cannot write a character: the fallback translator,
//! and the mark left when there is none.
//!
//! Every test here needs a model and is skipped without it: NLLB from
//! `LI_MT_MODEL_DIR`, Opus-MT from `LI_MT_FALLBACK_DIR`, each defaulting to
//! the shared model cache.

use std::path::PathBuf;

use li_mt::{LocalNllb, NllbConfig, chunk::Marks, opus::LocalOpus};

fn nllb_dir() -> Option<PathBuf> {
    let d = std::env::var("LI_MT_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| li_mt::nllb::default_model_dir());
    d.join("model.bin").is_file().then_some(d)
}

fn opus_dir() -> Option<PathBuf> {
    let d = std::env::var("LI_MT_FALLBACK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| li_mt::opus::default_model_dir());
    d.join("model.bin").is_file().then_some(d)
}

/// 饋 is not in NLLB's vocabulary, so "feedback" (反饋) comes back with
/// `<unk>` where it should be. Decoding used to drop the token and print 反
/// -- a different word -- instead of an evidently missing one.
#[test]
fn a_character_nllb_cannot_write_is_marked_rather_than_dropped() {
    let Some(dir) = nllb_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        ..Default::default()
    })
    .unwrap();

    let got = mt
        .translate_blocking("So I'd really like to get your feedback.", Marks::Heard)
        .unwrap();

    assert!(got.contains("反□"), "{got}");
}

#[test]
fn a_character_nllb_cannot_write_is_written_by_the_fallback() {
    let (Some(dir), Some(fallback)) = (nllb_dir(), opus_dir()) else {
        eprintln!("skipped: needs both LI_MT_MODEL_DIR and LI_MT_FALLBACK_DIR");
        return;
    };
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        fallback_model_dir: Some(fallback),
        ..Default::default()
    })
    .unwrap();

    let got = mt
        .translate_blocking("So I'd really like to get your feedback.", Marks::Heard)
        .unwrap();

    assert!(got.contains("反饋"), "{got}");
}

/// The fallback is a separate download. Asked for and not there, NLLB still
/// runs, and marks the gaps.
#[test]
fn a_fallback_that_is_not_installed_leaves_nllb_working() {
    let Some(dir) = nllb_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        fallback_model_dir: Some("/nonexistent/opus".into()),
        ..Default::default()
    })
    .unwrap();

    let got = mt
        .translate_blocking("So I'd really like to get your feedback.", Marks::Heard)
        .unwrap();

    assert!(got.contains("反□"), "{got}");
}

/// `li_mt::spm` rebuilds SentencePiece's tokenizer from the model file rather
/// than linking the C++ library. These are SentencePiece's own pieces for the
/// same text (sentencepiece 0.2, 2026-09-24): full-width letters and accents
/// are where a normaliser that is only nearly right shows.
#[test]
fn the_fallback_reads_its_source_exactly_as_sentencepiece_does() {
    let Some(dir) = opus_dir() else {
        eprintln!("skipped: no fallback MT model; set LI_MT_FALLBACK_DIR");
        return;
    };
    let opus = LocalOpus::open(&dir, 1).unwrap();

    let got = opus
        .source_pieces("Ｆｕｌｌｗｉｄｔｈ ＡＢＣ and café naïve")
        .unwrap();

    assert_eq!(
        got,
        [
            "▁Full", "wi", "d", "th", "▁A", "BC", "▁and", "▁caf", "é", "▁na", "ï", "ve"
        ]
    );
}
