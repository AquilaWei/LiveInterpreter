//! Turning NLLB's output into something a Taiwanese reader would accept.
//!
//! Two steps, both required, both measured:
//!
//! * **OpenCC `s2twp`.** NLLB is asked for Simplified Chinese (see
//!   `nllb::TGT_LANG` for why), so this makes the characters Traditional and
//!   the *words* Taiwanese in one pass: 项目经理 becomes 專案經理, 软件 軟體,
//!   信息 訊息, 鼠标 滑鼠. The product calls Traditional Chinese a hard
//!   requirement, and the character set alone would not satisfy it. It is
//!   also cheap enough not to matter: a few microseconds a line against a
//!   quarter of a second for the translation (measured 2026-09-24, which is
//!   also when two faster converters were tried and dropped -- `zhconv` leaves
//!   數據 and 信息 alone, and `opencc-fmmseg` only buys 3 µs).
//! * **Full-width punctuation.** NLLB emits ASCII `,` `.` `:` with a space
//!   after, so a subtitle line reads "我是薩拉, 專案經理." rather than
//!   "我是薩拉，專案經理。". Rewriting it lifted chrF against the zh-TW
//!   references by 2.3 and 2.2 points on the two clips, for a regex.
//!
//! The converter is not `opencc-rust`: see the workspace `Cargo.toml`. Whether
//! the substitution is faithful is not taken on trust -- `tests/reference.rs`
//! replays OpenCC 1.1.9's own output for every case in the prototype's fixture.

use anyhow::{Context, Result};
use ferrous_opencc::{OpenCC, config::BuiltinConfig};

pub struct Zh {
    cc: OpenCC,
}

impl Zh {
    pub fn new() -> Result<Self> {
        let cc = OpenCC::from_config(BuiltinConfig::S2twp)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("loading the OpenCC s2twp conversion chain")?;
        Ok(Self { cc })
    }

    /// Mainland vocabulary to Taiwan vocabulary.
    pub fn to_tw(&self, s: &str) -> String {
        self.cc.convert(s)
    }

    /// Join the per-chunk translations of one line and make it a subtitle.
    pub fn finish(&self, parts: &[String]) -> String {
        let mut joined = String::new();
        for p in parts.iter().map(|p| p.trim()).filter(|p| !p.is_empty()) {
            if !joined.is_empty() && !ends_open(&joined) {
                joined.push(' ');
            }
            joined.push_str(p);
        }
        let converted = self.to_tw(&joined);
        punctuate(&converted)
    }
}

/// Does the text so far end on punctuation, so the next chunk can abut it?
fn ends_open(s: &str) -> bool {
    matches!(
        s.chars().next_back(),
        Some(',' | '.' | ':' | ';' | '!' | '?' | '，' | '。' | '：' | '；' | '！' | '？')
    )
}

/// ASCII punctuation to full-width, and the spaces around it away.
///
/// Everything is rewritten except a mark wedged *inside* a Latin token, which
/// is where a translation that kept an English fragment holds its decimal
/// points and abbreviations: "3.5 GHz" must not become "3。5 GHz".
fn punctuate(s: &str) -> String {
    let ch: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < ch.len() {
        let c = ch[i];
        if c == '.' {
            // A run of two or more is an ellipsis, which Chinese writes as ……
            let mut j = i;
            while j < ch.len() && ch[j] == '.' {
                j += 1;
            }
            if j - i >= 2 {
                trim_space(&mut out);
                out.push_str("……");
                i = j;
                skip_space(&ch, &mut i);
                continue;
            }
        }
        if let Some(full) = full_width(c)
            && !inside_a_latin_token(&ch, i)
        {
            trim_space(&mut out);
            out.push(full);
            i += 1;
            skip_space(&ch, &mut i);
            continue;
        }
        if c.is_whitespace() && is_han(prev_non_space(&out)) && is_han(next_non_space(&ch, i + 1)) {
            // NLLB inserts a space at every sub-word boundary it decoded.
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    // A line that ends mid-clause is NLLB giving up, not the speaker pausing.
    out.trim().trim_end_matches(['，', '、', ' ']).to_string()
}

fn full_width(c: char) -> Option<char> {
    Some(match c {
        ',' => '，',
        '.' => '。',
        ':' => '：',
        ';' => '；',
        '!' => '！',
        '?' => '？',
        _ => return None,
    })
}

/// True when the mark has an ASCII letter or digit hard against it on both
/// sides -- "3.5", "e.g", "a,b" -- and so belongs to the token, not the
/// sentence. A space on either side means it is punctuation.
fn inside_a_latin_token(ch: &[char], i: usize) -> bool {
    let alnum = |c: Option<&char>| c.is_some_and(|c| c.is_ascii_alphanumeric());
    alnum(ch.get(i.wrapping_sub(1))) && alnum(ch.get(i + 1))
}

fn is_han(c: Option<char>) -> bool {
    matches!(c, Some(c) if ('\u{4e00}'..='\u{9fff}').contains(&c)
        || ('\u{3400}'..='\u{4dbf}').contains(&c)
        || ('\u{f900}'..='\u{faff}').contains(&c))
}

fn prev_non_space(s: &str) -> Option<char> {
    s.chars().rev().find(|c| !c.is_whitespace())
}

fn next_non_space(ch: &[char], from: usize) -> Option<char> {
    ch[from..].iter().find(|c| !c.is_whitespace()).copied()
}

fn trim_space(out: &mut String) {
    while out.ends_with(' ') {
        out.pop();
    }
}

fn skip_space(ch: &[char], i: &mut usize) {
    while *i < ch.len() && ch[*i] == ' ' {
        *i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zh() -> Zh {
        Zh::new().unwrap()
    }

    #[test]
    fn mainland_vocabulary_becomes_taiwan_vocabulary() {
        let z = zh();
        assert_eq!(z.to_tw("項目經理"), "專案經理");
        assert_eq!(z.to_tw("软件"), "軟體");
        assert_eq!(z.to_tw("鼠标"), "滑鼠");
        assert_eq!(z.to_tw("内存"), "記憶體");
    }

    #[test]
    fn punctuation_becomes_full_width_and_the_spaces_go() {
        assert_eq!(punctuate("我是薩拉, 專案經理."), "我是薩拉，專案經理。");
        assert_eq!(
            punctuate("這樣的話: 我們就開始吧!"),
            "這樣的話：我們就開始吧！"
        );
    }

    #[test]
    fn a_latin_fragment_keeps_its_own_decimal_point() {
        assert_eq!(
            punctuate("頻率是 3.5 GHz, 不是 2.4."),
            "頻率是 3.5 GHz，不是 2.4。"
        );
    }

    #[test]
    fn an_ellipsis_becomes_the_chinese_one() {
        assert_eq!(punctuate("或我們可以..."), "或我們可以……");
    }

    #[test]
    fn a_line_left_hanging_on_a_comma_is_closed() {
        assert_eq!(punctuate("我們會做一些事情,"), "我們會做一些事情");
    }

    #[test]
    fn chunks_are_joined_without_a_seam() {
        let z = zh();
        assert_eq!(
            z.finish(&[
                "她与蕾蒂的重逢温柔得难以言喻,".into(),
                "接下来的日子...".into()
            ]),
            "她與蕾蒂的重逢溫柔得難以言喻，接下來的日子……"
        );
        assert_eq!(z.finish(&[]), "");
        assert_eq!(z.finish(&["  ".into()]), "");
    }
}
