//! Interjections: the pieces NLLB has nothing to translate in, and so makes
//! something up for.
//!
//! Handed "Mm-hmm." or "Oh." alone, NLLB does not answer 嗯 or 喔. It answers
//! with the most ordinary sentence it knows, and for this model that is
//! 沒有任何問題 ("no problem at all"). In a 24-minute conversation
//! (2026-09-23) it did so 18 times, each one putting words in a speaker's
//! mouth. So a piece made only of interjections is not sent to the model: each
//! word becomes its Chinese counterpart, and the piece keeps its punctuation.
//!
//! ## How a word is recognised
//!
//! Not by a list of spellings. Speech gets written "Um", "Umm", "Ummm", "Uhm",
//! "Mm-hmm", "Mmhmm", "Mhm", "Hmmm", and a list is always one spelling short.
//! Not by edit distance either: "on", "no", "so" and "ok" are each one edit
//! from "oh", and a match that swallows a real word loses content, which is
//! worse than the fault being fixed.
//!
//! A word is instead reduced to its shape -- lower case, marks removed,
//! repeated letters collapsed, so "Ummm" is "um" and "Mmm-hmm" is "m-hm" --
//! and the shape is classified by which letters it is made of: only `h` and
//! `m` is a hum, `u` followed by `h`/`m` is a hesitation, `o` followed by
//! `o`/`h` is "oh". Any spelling a transcriber invents inside those shapes is
//! covered, and no English word fits them. The handful of interjections that
//! are real words ("yeah", "okay", "right", "wow") are looked up after the same
//! reduction, so "Yeahhh" and "Okayyy" still count.
//!
//! ## Cost
//!
//! One pass over each word, and the first word that is not an interjection
//! ends it. An ordinary sentence costs a few character comparisons.

/// The Chinese for `piece` when every word in it is an interjection; `None`
/// when any word is not, and the piece should go to the model.
///
/// Punctuation stays where it was ("Oh, yeah." gives "喔,對."), for
/// [`crate::Zh::finish`] to make full-width like any translation. A piece
/// with no words at all is `None`: there is nothing here to answer for it.
pub fn render(piece: &str) -> Option<String> {
    let mut out = String::new();
    let mut any = false;
    for token in piece.split_whitespace() {
        let core = token.trim_matches(|c: char| !c.is_alphanumeric());
        if core.is_empty() {
            // A dialogue dash or a lone ellipsis: nothing to translate.
            continue;
        }
        out.push_str(interjection(core)?);
        any = true;
        let head = token.trim_end_matches(|c: char| !c.is_alphanumeric());
        out.push_str(&token[head.len()..]);
    }
    any.then_some(out)
}

/// The Chinese for one word, if it is an interjection.
fn interjection(word: &str) -> Option<&'static str> {
    let w = shape(word)?;
    if w.contains('-') {
        return compound(&w);
    }
    match w.as_str() {
        "yeah" | "yea" | "yeh" | "yah" | "ya" | "yep" | "yup" | "yes" => Some("對"),
        // A word with other meanings, but a clause of nothing else is a tag
        // question or a nod: "Right?" came back as 沒有任何問題 three times
        // in the conversation this was written for.
        "right" => Some("對"),
        "okay" | "okey" | "ok" => Some("好"),
        "huh" => Some("蛤"),
        "wow" => Some("哇"),
        "eh" => Some("欸"),
        _ => sound(&w),
    }
}

/// "Mm-hmm", "uh-huh", "uh-uh": interjections written in parts.
fn compound(w: &str) -> Option<&'static str> {
    match w {
        "uh-huh" => Some("嗯"),
        "uh-uh" => Some("不"),
        // "mm-hmm", "hmm-mm", "mm-mm": every part a hum.
        _ if w.split('-').all(|p| sound(p) == Some("嗯")) => Some("嗯"),
        _ => None,
    }
}

/// Classify a shape by the letters it is made of. `w` is already reduced:
/// lower case, no repeated letters.
fn sound(w: &str) -> Option<&'static str> {
    let made_of = |s: &str, set: &str| !s.is_empty() && s.chars().all(|c| set.contains(c));
    let (first, rest) = w.split_at(w.chars().next()?.len_utf8());
    if made_of(w, "hm") && w.contains('m') {
        // hm, mm, mhm, hmm: a hum is 嗯 whichever way round it is spelled.
        return Some("嗯");
    }
    match first {
        // oh, ooh
        "o" if made_of(rest, "oh") && rest.contains('h') => Some("喔"),
        // ah, aah, aha
        "a" if made_of(rest, "ah") && rest.contains('h') => Some("啊"),
        // uh, um, uhm
        "u" if made_of(rest, "hm") => Some("呃"),
        // er, erm, ehm. "e" only once, so "ere" is not one.
        "e" if made_of(rest, "rhm") => Some("呃"),
        _ => None,
    }
}

/// Lower case, marks dropped, runs of a letter collapsed to one. Hyphens are
/// kept, one at a time, because "uh-uh" and "uh-huh" mean opposite things.
/// `None` for anything with a digit or a letter outside ASCII in it: no
/// interjection has either.
fn shape(word: &str) -> Option<String> {
    let mut out = String::new();
    for c in word.chars() {
        let c = c.to_ascii_lowercase();
        match c {
            'a'..='z' | '-' if out.ends_with(c) => {}
            'a'..='z' => out.push(c),
            '-' if !out.is_empty() => out.push(c),
            // "o.k.", "'kay"
            '.' | '\'' | '’' => {}
            _ => return None,
        }
    }
    let out = out.trim_end_matches('-');
    (!out.is_empty()).then(|| out.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hum_is_rendered_as_its_chinese_with_the_punctuation_kept() {
        assert_eq!(render("Mm-hmm.").as_deref(), Some("嗯."));
    }

    #[test]
    fn a_hum_with_extra_letters_is_still_a_hum() {
        assert_eq!(render("Mmmm-hmmmm").as_deref(), Some("嗯"));
    }

    #[test]
    fn a_hum_written_as_one_word_is_still_a_hum() {
        assert_eq!(render("Mhm").as_deref(), Some("嗯"));
    }

    #[test]
    fn a_drawn_out_hesitation_is_a_hesitation() {
        assert_eq!(render("Ummm...").as_deref(), Some("呃..."));
    }

    #[test]
    fn a_hesitation_spelled_with_both_letters_is_a_hesitation() {
        assert_eq!(render("Uhm,").as_deref(), Some("呃,"));
    }

    #[test]
    fn a_drawn_out_oh_is_an_oh() {
        assert_eq!(render("Ohhh!").as_deref(), Some("喔!"));
    }

    #[test]
    fn a_drawn_out_yeah_is_a_yeah() {
        assert_eq!(render("Yeahhh.").as_deref(), Some("對."));
    }

    #[test]
    fn right_on_its_own_is_a_tag_question() {
        assert_eq!(render("Right?").as_deref(), Some("對?"));
    }

    #[test]
    fn right_inside_a_clause_goes_to_the_model() {
        assert_eq!(render("Turn right."), None);
    }

    #[test]
    fn several_interjections_keep_their_own_punctuation() {
        assert_eq!(render("Oh, yeah. Okay.").as_deref(), Some("喔,對.好."));
    }

    #[test]
    fn a_dialogue_dash_is_not_a_word() {
        assert_eq!(render("- Mm-hmm.").as_deref(), Some("嗯."));
    }

    #[test]
    fn agreement_and_refusal_written_alike_are_told_apart() {
        assert_eq!(render("Uh-huh.").as_deref(), Some("嗯."));
        assert_eq!(render("Uh-uh.").as_deref(), Some("不."));
    }

    #[test]
    fn a_piece_with_one_real_word_goes_to_the_model() {
        assert_eq!(render("Yeah, so..."), None);
    }

    #[test]
    fn a_word_one_letter_from_oh_is_not_an_oh() {
        assert_eq!(render("On."), None);
        assert_eq!(render("No."), None);
        assert_eq!(render("So."), None);
    }

    #[test]
    fn a_unit_made_of_hum_letters_is_not_a_hum_next_to_a_number() {
        assert_eq!(render("5 mm"), None);
    }

    #[test]
    fn a_word_made_of_the_same_letters_as_a_hesitation_is_not_one() {
        assert_eq!(render("Ohm."), None);
        assert_eq!(render("Hum."), None);
        assert_eq!(render("Her."), None);
    }

    #[test]
    fn a_piece_of_nothing_but_marks_is_left_to_the_model() {
        assert_eq!(render("..."), None);
    }

    #[test]
    fn the_rendered_piece_reads_as_a_chinese_subtitle() {
        let zh = crate::Zh::new().unwrap();

        assert_eq!(zh.finish(&[render("Oh, yeah.").unwrap()]), "喔，對。");
    }
}
