//! Translation, English to Traditional Chinese (Taiwan).
//!
//! Most open MT models emit Simplified Chinese vocabulary even when asked for
//! Traditional, so a backend that cannot promise Taiwan usage says so via
//! [`Translator::target_is_zh_tw`] and post-processes with OpenCC `s2twp`
//! (软件 -> 軟體). An LLM backend prompted for Taiwanese Traditional answers
//! `true` and is passed through untouched.
//!
//! ## What was measured
//!
//! The Python prototype's MT path was a single call per finalised line. On the
//! accurate lane's real output that turned out to lose about a third of the
//! content: NLLB-200-distilled-600M is a sentence-level model, and a line
//! carrying two clauses comes back with one of them translated and the other
//! silently gone. Nothing in the decoder fixes it. Cutting the line at its own
//! punctuation first does:
//!
//! | | chrF `read_clean` | chrF `read_hard` | lines cut short |
//! |---|---|---|---|
//! | one call per line, beam 1 (the prototype's settings) | 9.9 | 15.2 | 70% / 18% |
//! | one call per line, beam 4 | 12.2 | 15.2 | 60% / 12% |
//! | **clause-split at >6 words, beam 4** | **19.7** | **17.6** | **20% / 0%** |
//!
//! chrF is against `testdata/*.zh.txt`, which is machine-written and unverified
//! (see `testdata/README.md`), so the *differences* are the result and the
//! absolute values are not quotable. "Lines cut short" -- the share of
//! translations ending mid-clause on a comma -- needs no reference at all, and
//! is the number the design is built on.
//!
//! Pipeline, per finalised line: [`chunk::split`] -> encode -> CTranslate2 ->
//! decode -> [`zh::Zh::finish`] (OpenCC `s2twp`, then punctuation). Measured
//! over the four golden clips at 195-368 ms a line, 770-810 MiB peak.

use anyhow::Result;
use async_trait::async_trait;

pub mod chunk;
pub mod filler;
pub mod nllb;
pub mod zh;

pub use nllb::{LocalNllb, NllbConfig};
pub use zh::Zh;

#[async_trait]
pub trait Translator: Send + Sync {
    /// `ctx` carries the previous sentences; subtitles read far better when the
    /// translator can see what came before.
    ///
    /// A backend that cannot use context ignores it -- and the local NLLB
    /// backend is one, measurably so; see [`nllb`]. `li-core` still keeps the
    /// history, because whether it helps is a property of the backend and the
    /// backend can be swapped at run time.
    async fn translate(&self, src: &str, ctx: &[String]) -> Result<String>;

    /// False means the caller post-processes with OpenCC.
    fn target_is_zh_tw(&self) -> bool;
}
