//! Checks this crate against the Python prototype it replaces, and the
//! translations that went wrong in real use.
//!
//! `nllb_reference.json` was produced by running the prototype's `mt.py`
//! unchanged over 53 lines -- real accurate-lane output from an eval run,
//! promoted fast-lane text, technical sentences that separate zh-CN from zh-TW
//! usage, and the degenerate fragments `li-stream` really does hand over. It
//! records, for each: the token sequence transformers built, the hypothesis
//! CTranslate2 returned, the text before OpenCC, and the text after it.
//!
//! The OpenCC checks need no model and run in CI. The rest need the 600 MB
//! model and are skipped when it is not there, exactly like the ASR
//! integration tests.
//!
//! The prototype asked NLLB for `zho_Hant`; this crate now asks for
//! `zho_Hans` (see `nllb::TGT_LANG`). The tokens it sends are unchanged, so
//! they are still checked against the prototype's. Its translations are not:
//! they were meant to differ, and the regressions below say what for.

use std::path::PathBuf;

use li_mt::{LocalNllb, NllbConfig, Zh};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    opencc_profile: String,
    src_lang: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    src: String,
    /// After OpenCC. Absent content means the source was blank.
    zh: String,
    /// Before OpenCC. Missing for blank sources.
    raw: Option<String>,
    src_tokens: Option<Vec<String>>,
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/nllb_reference.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("valid fixture")
}

/// The reason `ferrous-opencc` is used instead of a binding to libopencc: it
/// has to agree with OpenCC 1.1.9 first.
#[test]
fn the_pure_rust_opencc_matches_opencc_1_1_9_on_every_case() {
    let f = fixture();
    assert_eq!(f.opencc_profile, "s2twp");
    let zh = Zh::new().unwrap();
    let mut checked = 0;
    for c in &f.cases {
        let Some(raw) = &c.raw else { continue };
        assert_eq!(zh.to_tw(raw), c.zh, "s2twp disagreed on {raw:?}");
        checked += 1;
    }
    assert!(
        checked >= 45,
        "only {checked} cases carried a pre-OpenCC string"
    );
}

/// The whole point of the conversion: even NLLB's Traditional output (the
/// fixture's `zho_Hant`) is not Taiwan usage.
#[test]
fn the_conversion_is_not_a_no_op() {
    let f = fixture();
    let changed = f
        .cases
        .iter()
        .filter(|c| c.raw.as_ref().is_some_and(|r| r != &c.zh))
        .count();
    assert!(
        changed >= 5,
        "s2twp changed only {changed} of the reference cases"
    );
}

fn model_dir() -> Option<PathBuf> {
    let d = std::env::var("LI_MT_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| li_mt::nllb::default_model_dir());
    d.join("model.bin").is_file().then_some(d)
}

/// Tokenisation is where a port silently diverges: this model ships a
/// `tokenizer.json` whose post-processor is the pre-fix NLLB template, and
/// getting the language token wrong costs quality without erroring.
#[test]
fn the_source_tokens_match_transformers_exactly() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let f = fixture();
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        ..Default::default()
    })
    .unwrap();
    let mut checked = 0;
    for c in &f.cases {
        let Some(want) = &c.src_tokens else { continue };
        let got = mt.source_tokens(&c.src).unwrap();
        assert_eq!(&got, want, "tokens differ for {:?}", c.src);
        assert_eq!(got.first().unwrap(), &f.src_lang);
        checked += 1;
    }
    assert!(checked >= 45);
}

/// "Hello everybody." is how a meeting starts, and asked for `zho_Hant` NLLB
/// answered it with a web page's breadcrumb, 您的位置: 首頁 ("you are here:
/// home"). It must come back as a greeting.
#[test]
fn a_greeting_is_not_translated_into_a_web_page_breadcrumb() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        ..Default::default()
    })
    .unwrap();

    let got = mt
        .translate_blocking("Hello everybody.", li_mt::chunk::Marks::Heard)
        .unwrap();

    assert!(!got.contains("您的位置"), "{got}");
}

/// "Oh, well." has a real word in it, so the interjection table does not
/// catch it, and asked for `zho_Hant` NLLB answered 沒有任何問題 ("no problem
/// at all") -- a sentence nobody said.
#[test]
fn a_shrug_is_not_translated_into_no_problem_at_all() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        ..Default::default()
    })
    .unwrap();

    let got = mt
        .translate_blocking("Oh, well.", li_mt::chunk::Marks::Heard)
        .unwrap();

    assert!(!got.contains("沒有任何問題"), "{got}");
}

/// Whisper writes a silence as a row of spaced dots, and each dot used to reach
/// the model as a sentence of its own: 沒有人知道 ("nobody knows"), once per
/// dot, fifteen lines of it in one meeting.
#[test]
fn a_silence_written_as_dots_is_not_translated_into_words() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        ..Default::default()
    })
    .unwrap();

    let got = mt
        .translate_blocking(". . . . . .", li_mt::chunk::Marks::Heard)
        .unwrap();

    assert_eq!(got, "……");
}

/// The same dots inside a sentence: the pause is kept, and nothing is
/// invented for it. Before, each dot came back as a sentence of its own --
/// 沒有人知道 when NLLB was asked for Traditional, 讓我們一起去 ("let's go
/// together") since it is asked for Simplified -- so what is pinned is the
/// text around the dots, not the absence of any one invention.
#[test]
fn a_pause_inside_a_sentence_is_not_translated_into_words() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        ..Default::default()
    })
    .unwrap();

    let got = mt
        .translate_blocking(
            "Other logistics? Uh . . . Oh, we also . . . we also are bringing up treat times.",
            li_mt::chunk::Marks::Heard,
        )
        .unwrap();

    // Only the part the dots were in is pinned: the last clause's wording
    // differs between a debug and a release build of CTranslate2 (提及 against
    // 提到), which is rounding, not this.
    assert!(got.starts_with("其他物流？呃我們也……"), "{got}");
}

/// Alone, "Mm-hmm." came back from the model as 沒有任何問題。 ("no problem at
/// all"), 18 times in one 24-minute conversation. It must not reach the model.
#[test]
fn a_hum_on_its_own_is_not_translated_into_a_sentence() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        ..Default::default()
    })
    .unwrap();

    let got = mt
        .translate_blocking("Mm-hmm.", li_mt::chunk::Marks::Heard)
        .unwrap();

    assert_eq!(got, "嗯。");
}
