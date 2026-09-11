//! Word-level text handling shared by the commit policy and the merge guard.
//!
//! Both lanes hand us [`Word`]s whose `text` carries no leading space: the fast
//! lane strips sherpa's `\u{2581}` marker and whisper's sub-word pieces are
//! already glued together by `li-asr::words::merge`. So a line is the words
//! joined by single spaces, and punctuation, which never opens a word, stays
//! attached to the token before it.

use li_types::Word;

/// Words that legitimately vanish between the lanes. The fast lane transcribes
/// every backchannel it hears; whisper is trained to leave them out, and a
/// subtitle is better without them. Comparing raw word counts would read that
/// as a dropped clause -- the same distinction the G2a/G2b metrics draw.
const FILLER: &[&str] = &[
    "um", "uh", "er", "erm", "ah", "eh", "huh", "hmm", "mm", "mhm", "mmm", "uhhuh", "mmhmm",
    "yeah", "yep", "yup", "okay", "ok", "kay",
];

/// Lower-case, with leading and trailing punctuation removed.
///
/// Used to compare the two lanes and to compare consecutive hypotheses, neither
/// of which should care that whisper writes "menu," where sherpa writes "MENU".
pub fn normalize(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
        .to_lowercase()
}

pub fn join(words: &[Word]) -> String {
    let mut out = String::new();
    for w in words {
        let t = w.text.trim();
        if t.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(t);
    }
    out
}

/// The normalised words of `s` with fillers and empties removed.
pub fn content_words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty() && !FILLER.contains(&w.as_str()))
        .collect()
}

/// Does `words` contain the same run of `n` words twice?
///
/// Only ever asked of text that is already known to be much longer than what
/// the other lane heard over the same audio, so one repeat is enough evidence.
/// Asked on its own it would fire on ordinary English ("i think that" twice in
/// a paragraph is unremarkable), which is why [`crate::merge::decide`] gates it
/// behind the length ratio rather than using it alone.
pub fn has_repeated_run(words: &[String], n: usize) -> bool {
    if n == 0 || words.len() < 2 * n {
        return false;
    }
    let mut seen = std::collections::HashSet::new();
    words.windows(n).any(|w| !seen.insert(w.join(" ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn w(text: &str) -> Word {
        Word {
            text: text.into(),
            start: Duration::ZERO,
            end: Duration::ZERO,
        }
    }

    #[test]
    fn punctuation_stays_attached_to_its_word() {
        assert_eq!(join(&[w("the"), w("menu,"), w("and")]), "the menu, and");
    }

    #[test]
    fn normalising_ignores_case_and_edge_punctuation_but_keeps_apostrophes() {
        assert_eq!(normalize("MENU,"), "menu");
        assert_eq!(normalize("\"Don't\""), "don't");
    }

    #[test]
    fn fillers_are_not_content() {
        assert_eq!(content_words("um so yeah the menu"), ["so", "the", "menu"]);
    }

    #[test]
    fn a_repeated_run_is_found_and_ordinary_text_is_not() {
        let looped = content_words(
            "and then when you go on the menu you can select the description box \
             and then when you go on the menu you can select the description box",
        );
        assert!(has_repeated_run(&looped, 4));

        let plain =
            content_words("this is our first meeting surprisingly enough this is our agenda");
        // "this is our" repeats, but not four words in a row.
        assert!(!has_repeated_run(&plain, 4));
    }
}
