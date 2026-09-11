//! Checks this crate against the Phase 0 Python implementation it replaces.
//!
//! `nllb_reference.json` was produced by running `poc/liveinterpreter_poc/mt.py`
//! unchanged over 53 lines -- real accurate-lane output from the task 1.5 eval,
//! promoted fast-lane text, technical sentences that separate zh-CN from zh-TW
//! usage, and the degenerate fragments `li-stream` really does hand over. It
//! records, for each: the token sequence transformers built, the hypothesis
//! CTranslate2 returned, the text before OpenCC, and the text after it.
//!
//! Two of the three checks need no model and run in CI. The third needs the
//! 600 MB model and is skipped when it is not there, exactly like the ASR
//! integration tests.

use std::path::PathBuf;

use li_mt::{LocalNllb, NllbConfig, Zh};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    opencc_profile: String,
    src_lang: String,
    tgt_lang: String,
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

/// The whole point of the conversion: `zho_Hant` is not Taiwan usage.
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

/// Phase 0's settings (greedy, one call per line, no trimming) against Phase
/// 0's output.
///
/// This does **not** assert equality, and the reason is worth stating: the int8
/// GEMM compiled in here is ruy, and the PyPI `ctranslate2` wheel's is oneDNN.
/// They round differently in the last bits, which is enough to flip the argmax
/// wherever two tokens are near-tied -- and greedy decoding then never comes
/// back. About three fifths of the lines still match; the rest differ in ways
/// that read as neither better nor worse ("我們有25分鐘時間去做," against
/// "我們有25分鐘時間去做這件事."). What *is* asserted exactly is the token
/// sequence, above, which is where a port actually goes wrong.
///
/// So the bar here is a smoke test: a wrong language token or a mangled
/// post-processor would send this to nearly zero, not to 61%. It is capped at
/// the first `SAMPLE` lines because a debug build of CTranslate2 decodes at
/// about five seconds a line, and the rate is what is being read, not the
/// count.
#[test]
fn phase_0_settings_reproduce_phase_0_output() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no MT model; set LI_MT_MODEL_DIR");
        return;
    };
    let f = fixture();
    assert_eq!(f.tgt_lang, "zho_Hant");
    let mt = LocalNllb::open(&NllbConfig {
        model_dir: dir,
        beam_size: 1,
        max_chunk_words: 0,
        trim_final_stop: false,
        ..Default::default()
    })
    .unwrap();
    const SAMPLE: usize = 20;
    let (mut same, mut total) = (0, 0);
    for c in &f.cases {
        let Some(raw) = &c.raw else { continue };
        if total == SAMPLE {
            break;
        }
        total += 1;
        let got = mt.pieces(&c.src, li_mt::chunk::Marks::Heard).unwrap();
        assert_eq!(got.len(), 1, "no split was asked for: {:?}", c.src);
        if &got[0] == raw {
            same += 1;
        } else {
            eprintln!("differs: {:?}\n  py {raw:?}\n  rs {:?}", c.src, got[0]);
        }
    }
    let agreement = same as f64 / total as f64;
    eprintln!(
        "reproduced {same}/{total} Phase 0 lines ({:.0}%)",
        100.0 * agreement
    );
    assert!(
        agreement >= 0.6,
        "only {same}/{total} lines reproduced Phase 0"
    );
}
