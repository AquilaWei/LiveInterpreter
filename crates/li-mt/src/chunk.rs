//! Cutting a finalised line into pieces NLLB can actually translate.
//!
//! NLLB-200 is a *sentence-level* model. Handed a line with more than one
//! clause it routinely translates the first one and stops -- measured on the
//! accurate-lane output, **33% of lines came back ending in a comma**,
//! mid-clause, and no decoding parameter moved that number (beam 1/2/4/5,
//! length penalty 1.0/2.0, `no_repeat_ngram_size`: all 33-41%). Cutting the
//! line at its own punctuation first and translating the pieces separately took
//! it to 7.4%, and chrF against the zh-TW reference from 8.7 to 16.4 on
//! `read_clean`.
//!
//! The cut is preferably made at punctuation the speaker's own words carry, so
//! a piece is usually a clause rather than a fixed-width window.
//!
//! ## Why there is a width cut at all
//!
//! This used to end "a line with no internal punctuation comes back whole
//! however long it is, which is the right failure: a bad translation of a whole
//! clause beats a fluent translation of half of one". That was written when the
//! only lines reaching the translator came from the accurate lane, which
//! punctuates. The fast lane does not punctuate anything, so for it the rule
//! meant no cut ever, and `max_line_words = 40` is then the length NLLB is
//! handed. Measured on `read_clean`'s reference text cut into 40-word lines --
//! which is exactly what the fast lane produces -- the translation came back
//! **0.63 Chinese characters per English word, against 1.55 in the reference**:
//! well over a third of every long line was simply not translated. chrF was
//! 10.4 against 19.4 for the same clip's punctuated accurate-lane transcript
//! through the same harness.
//!
//! So a stretch with nowhere to cut is cut into equal parts of at most
//! [`NllbConfig::max_run_words`](crate::NllbConfig::max_run_words) words. Every
//! width tried beat leaving it whole; 6 came out best and is the same number as
//! `max_chunk_words`, which was not the expectation -- an arbitrary 6-word cut
//! is a fragment where a 6-word cut at a comma is a clause. It is a separate
//! setting because it can be turned off and because the two cuts are different
//! acts, not because the widths came out different. See
//! the width sweep.

use std::ops::Range;

/// Ends a sentence. A cut here is always right.
const SENTENCE: &[char] = &['.', '!', '?'];
/// Ends a clause. Only cut here when the sentence is too long to survive whole.
const CLAUSE: &[char] = &[',', ';', ':'];

/// Where a line's punctuation came from, and therefore whether it is allowed to
/// be the only place the line is cut.
///
/// The distinction did not exist before punctuation restoration, because there was only one
/// answer: the accurate lane heard its marks and the fast lane had none, so
/// "does this text contain a full stop" told you both things at once. Now the
/// fast lane has marks too, put there by a model reading the words -- and
/// measured on the user's own sessions, it is right 84% of the times it opens
/// its mouth and finds 58% of the boundaries that are there. Good enough to
/// read; not good enough to be trusted as the sole cut, and in any case its
/// clauses are longer than NLLB carries whole. See [`split`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marks {
    /// A recogniser heard them. Today's rule, unchanged: a line with any mark
    /// in it is cut only at its marks.
    Heard,
    /// A punctuation model inferred them, or there are none at all. Cut at the
    /// marks *and* by width.
    Restored,
}

/// Split `text` so that no piece is longer than `max_words`, cutting at
/// sentence ends first and clause ends only where that was not enough.
///
/// A piece with no punctuation left to cut at is cut on word boundaries after
/// `run_words`; `run_words == 0` leaves it whole, which is the behaviour of
/// every release before the width cut.
///
/// Pieces are trimmed and never empty. `max_words == 0` turns splitting off
/// entirely, which is how the prototype's behaviour is reproduced in tests.
pub fn split(text: &str, max_words: usize, run_words: usize, marks: Marks) -> Vec<&str> {
    if max_words == 0 {
        let whole = text.trim();
        return if whole.is_empty() {
            Vec::new()
        } else {
            vec![whole]
        };
    }
    // Heard marks turn the width cut off, because halving a genuine 14-word
    // clause measured worse than leaving it whole: chrF 19.4 -> 16.7 on
    // `read_clean`'s accurate-lane transcript, 17.4 -> 15.7 on `read_hard`'s.
    // So the accurate lane goes through here exactly as it did.
    //
    // Restored marks do not, and it took a measurement to believe it.
    // Turning the cut off for them -- which is what "the line now has real
    // punctuation" looked like it meant -- took the fast lane's translation
    // from **1.48 Chinese characters per English word to 1.08** on
    // `read_clean` and 1.45 -> 1.31 on `read_hard`, and doubled the lines
    // ending mid-clause. The restorer's clauses are simply longer than NLLB
    // carries: one 17-word clause goes in and its second half does not come
    // out. Cutting at the restored marks *and* by width is better than either
    // alone -- 13.6 -> 16.7 chrF at 1.53 zh/word on `read_clean`, 12.2 -> 14.4
    // at 1.46 on `read_hard`, better than today on both counts on both clips.
    let run_words = if marks == Marks::Heard && (text.contains(SENTENCE) || text.contains(CLAUSE)) {
        0
    } else {
        run_words
    };
    let mut out = Vec::new();
    for sent in ranges(text, SENTENCE) {
        let s = &text[sent.clone()];
        if words(s) <= max_words {
            push_run(&mut out, s, run_words);
            continue;
        }
        // Glue clauses back together until the group is long enough to stand on
        // its own. Translating "and then," alone produces a fragment that reads
        // worse than the run-on it came from.
        let mut group: Option<Range<usize>> = None;
        for clause in ranges(s, CLAUSE) {
            let g = group.get_or_insert(clause.clone());
            g.end = clause.end;
            if words(&s[g.clone()]) >= max_words {
                push_run(&mut out, &s[group.take().unwrap()], run_words);
            }
        }
        if let Some(g) = group {
            push_run(&mut out, &s[g], run_words);
        }
    }
    out
}

/// Push `piece`, cut into equal parts if it is longer than `run_words`.
///
/// This is the last resort and it shows: the cut lands wherever the count runs
/// out, so a piece can begin at "of" or end at "the". It is still better than
/// what it replaces, because what it replaces is NLLB translating the first
/// clause of a 40-word run and dropping the other thirty words.
fn push_run<'a>(out: &mut Vec<&'a str>, piece: &'a str, run_words: usize) {
    let total = words(piece);
    if run_words == 0 || total <= run_words {
        out.push(piece);
        return;
    }
    // Equal pieces rather than full ones and a stub. Cutting 14 words every 12
    // leaves "her brother," on its own, and NLLB translates a two-word fragment
    // as a two-word fragment; two pieces of seven both read as clauses.
    let parts = total.div_ceil(run_words);
    let (base, extra) = (total / parts, total % parts);
    let mut cut = base + usize::from(extra > 0);
    let (mut part, mut start, mut n, mut in_word) = (1usize, 0usize, 0usize, false);
    for (i, c) in piece.char_indices() {
        if c.is_whitespace() {
            in_word = false;
            continue;
        }
        if !in_word {
            in_word = true;
            if n == cut {
                // `start` is always a word start and `piece` came in trimmed,
                // so only trailing space can be there and the slice is never
                // empty.
                out.push(piece[start..i].trim_end());
                start = i;
                cut += base + usize::from(part < extra);
                part += 1;
            }
            n += 1;
        }
    }
    out.push(piece[start..].trim_end());
}

/// Byte ranges of `s` between marks, trimmed, with empties dropped.
///
/// A mark only ends a piece when whitespace or the end of the string follows
/// it, so "3.5 seconds" and "Or we can..." stay in one piece.
fn ranges(s: &str, marks: &[char]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if !marks.contains(&c) {
            continue;
        }
        let Some(&(next, ch)) = it.peek() else {
            continue;
        };
        if !ch.is_whitespace() {
            continue;
        }
        push_trimmed(&mut out, s, start..next);
        start = i + c.len_utf8();
    }
    push_trimmed(&mut out, s, start..s.len());
    out
}

fn push_trimmed(out: &mut Vec<Range<usize>>, s: &str, r: Range<usize>) {
    let piece = &s[r.clone()];
    let lead = piece.len() - piece.trim_start().len();
    let trail = piece.len() - piece.trim_end().len();
    if lead + trail < piece.len() {
        out.push(r.start + lead..r.end - trail);
    }
}

fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

/// NLLB's most reproducible single failure: a source ending in a full stop
/// pulls the decoder toward boilerplate it saw in web crawl. "Hello everybody."
/// comes back as 您的位置: 首頁 ("Your location: Home"); the same words without
/// the stop come back as 您好,所有人. Dropping the stop costs nothing on the
/// target side -- [`crate::zh`] rebuilds the punctuation anyway -- but it is
/// **off by default**, because across the two scored clips the aggregate was a
/// wash. See `NllbConfig::trim_final_stop`.
pub fn trim_final_stop(s: &str) -> &str {
    let t = s.trim_end();
    match t.strip_suffix('.') {
        // Not an ellipsis: "Or we can..." keeps its dots, which NLLB carries
        // through into the Chinese and a reader expects to see.
        Some(head) if !head.ends_with('.') && !head.is_empty() => head,
        _ => t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every test below this line was written about text a recogniser heard,
    /// which is what all of them were before punctuation restoration. Shadowing `split` here
    /// keeps them saying that, and keeps the diff that introduced [`Marks`]
    /// from silently changing what any of them assert.
    fn split(text: &str, max_words: usize, run_words: usize) -> Vec<&str> {
        super::split(text, max_words, run_words, Marks::Heard)
    }

    #[test]
    fn restored_marks_are_cut_at_and_then_cut_by_width() {
        // The measurement in one assertion. This is the fast lane's own
        // text with the punctuation model's marks on it: 21 words, one full
        // stop. Trusting the mark alone hands NLLB a 12-word piece and a
        // 9-word one, and gets back the first clause of each.
        let line = "Her meeting with Letty was indescribably tender and the days that followed.                     Were pretty equally divided between her and her brother";
        let heard = super::split(line, 6, 6, Marks::Heard);
        assert_eq!(heard.len(), 2, "heard marks are the only cut: {heard:?}");
        let restored = super::split(line, 6, 6, Marks::Restored);
        assert!(restored.len() > heard.len(), "{restored:?}");
        assert!(
            restored.iter().all(|p| p.split_whitespace().count() <= 6),
            "nothing longer than the width survives: {restored:?}"
        );
        // ...and the words are all still there, in order.
        assert_eq!(restored.join(" ").split_whitespace().count(), 21);
    }

    #[test]
    fn text_with_no_marks_at_all_is_cut_the_same_either_way() {
        // The accurate lane's behaviour is unchanged by 1.25, and that includes
        // the case where whisper returned a line with no punctuation in it:
        // there is nothing to trust, so both answers are the width cut.
        let line = "SO CHRISTIE TURNED A DEAF EAR TO HER PROPHETIC SOUL AND GAVE HERSELF UP";
        assert_eq!(
            super::split(line, 6, 6, Marks::Heard),
            super::split(line, 6, 6, Marks::Restored)
        );
    }

    #[test]
    fn zero_turns_splitting_off() {
        assert_eq!(
            split("Okay. Hello everybody.", 0, 0),
            ["Okay. Hello everybody."]
        );
        assert_eq!(split("  ", 0, 0), Vec::<&str>::new());
    }

    #[test]
    fn a_short_line_is_left_alone() {
        assert_eq!(
            split("Okay, this is our agenda.", 6, 0),
            ["Okay, this is our agenda."]
        );
    }

    #[test]
    fn sentences_are_cut_even_when_short() {
        assert_eq!(
            split("Okay. Hello everybody. I'm Sarah.", 6, 0),
            ["Okay.", "Hello everybody.", "I'm Sarah."]
        );
    }

    #[test]
    fn a_long_sentence_is_cut_at_its_clauses() {
        // The transcript line that measured worst: NLLB translated
        // "and the days that followed..." and dropped the rest.
        let line = "Her meeting with Letty was indescribably tender, and the days that \
                    followed were pretty equally divided between her and her brother, in \
                    nursing the one and loving the other.";
        assert_eq!(
            split(line, 6, 0),
            [
                "Her meeting with Letty was indescribably tender,",
                "and the days that followed were pretty equally divided between her and her \
                 brother,",
                "in nursing the one and loving the other."
            ]
        );
    }

    #[test]
    fn a_decimal_point_is_not_a_sentence_end() {
        assert_eq!(
            split("It costs 3.5 million and takes 2.5 years.", 4, 0),
            ["It costs 3.5 million and takes 2.5 years."]
        );
    }

    #[test]
    fn an_ellipsis_ends_one_piece_rather_than_three_empty_ones() {
        assert_eq!(
            split("Or we can... maybe not.", 6, 0),
            ["Or we can...", "maybe not."]
        );
    }

    #[test]
    fn a_run_on_with_nowhere_to_cut_comes_back_whole_only_with_the_width_cut_off() {
        let line = "and then when you go on the menu you can select the description box";
        assert_eq!(split(line, 6, 0), [line]);
        // 14 words at a width of 12 is two sevens, not a twelve and a stub.
        assert_eq!(
            split(line, 6, 12),
            [
                "and then when you go on the",
                "menu you can select the description box"
            ]
        );
    }

    // What the fast lane actually hands the translator: no punctuation anywhere
    // and `max_line_words` of them. Whole, this is the 85.7%-cut-short case in
    // the module docs.
    #[test]
    fn a_fast_lane_line_is_cut_into_equal_pieces() {
        let line = "HER MEETING WITH LETTY WAS INDESCRIBABLY TENDER AND THE DAYS THAT \
                    FOLLOWED WERE PRETTY EQUALLY DIVIDED BETWEEN HER AND HER BROTHER IN \
                    NURSING THE ONE AND LOVING THE OTHER";
        let pieces = split(line, 6, 12);
        let lens: Vec<usize> = pieces
            .iter()
            .map(|p| p.split_whitespace().count())
            .collect();
        assert_eq!(lens, [10, 10, 9]);
        assert_eq!(pieces.join(" "), line);
    }

    #[test]
    fn the_width_cut_does_not_touch_a_punctuated_line() {
        // The accurate lane's longest measured clause: 14 words with nowhere to
        // cut inside it. Cutting it is what cost 2.7 chrF, so it stays whole.
        let line = "Her meeting with Letty was indescribably tender, and the days that \
                    followed were pretty equally divided between her and her brother, in \
                    nursing the one and loving the other.";
        assert_eq!(split(line, 6, 6), split(line, 6, 0));
        // One unpunctuated word away from being cut into fives.
        assert_eq!(
            split("Dan is obviously a very good friend of mine.", 6, 6).len(),
            1
        );
    }

    #[test]
    fn the_width_cut_leaves_a_short_line_alone() {
        assert_eq!(
            split("Okay, this is our agenda.", 6, 12),
            ["Okay, this is our agenda."]
        );
        // Exactly at the width is still one piece.
        let twelve = "one two three four five six seven eight nine ten eleven twelve";
        assert_eq!(split(twelve, 6, 12), [twelve]);
    }

    #[test]
    fn an_uneven_run_spreads_the_remainder_over_the_first_pieces() {
        // 25 words, width 12 -> three pieces: 9, 8, 8.
        let line: String = (1..=25)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let pieces = split(&line, 6, 12);
        let lens: Vec<usize> = pieces
            .iter()
            .map(|p| p.split_whitespace().count())
            .collect();
        assert_eq!(lens, [9, 8, 8]);
        assert_eq!(pieces.join(" "), line);
    }

    #[test]
    fn trailing_clause_marks_do_not_produce_empty_pieces() {
        assert_eq!(split("Okay, ", 6, 0), ["Okay,"]);
        assert_eq!(split("   ", 6, 0), Vec::<&str>::new());
        assert_eq!(split("", 6, 0), Vec::<&str>::new());
    }

    #[test]
    fn the_final_stop_goes_but_an_ellipsis_stays() {
        assert_eq!(trim_final_stop("Hello everybody."), "Hello everybody");
        assert_eq!(trim_final_stop("Or we can..."), "Or we can...");
        assert_eq!(trim_final_stop("Okay,"), "Okay,");
        assert_eq!(trim_final_stop("."), ".");
    }
}
