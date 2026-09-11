//! Turning a backend's sub-word pieces into [`Word`]s.
//!
//! Both engines emit BPE pieces with a start time each, and both mark a word
//! boundary on the piece rather than between them — sherpa-onnx with U+2581,
//! whisper.cpp with an ordinary leading space. Everything else continues the
//! word before it, which is also the behaviour we want for punctuation: `,`
//! belongs to the word it follows, not to a word of its own.
//!
//! The part worth reading twice is how a word's `end` is set. A piece's
//! timestamp is its *start*; nothing in either engine's output says where a
//! word stops. The Phase 0 PoC gave each word the same start and end, and every
//! SRT block came out zero-length (see the note in `li-types`). So a word is
//! closed at the next word's start, and the last word of a segment at the
//! segment end the caller passes in.

use std::time::Duration;

use li_types::Word;

/// sherpa-onnx's word-start marker.
const WORD_MARK: char = '\u{2581}';

/// Does this piece begin a new word?
pub fn starts_word(piece: &str) -> bool {
    piece.starts_with(WORD_MARK) || piece.starts_with(' ')
}

/// Strip the boundary marker, whichever engine wrote it.
pub fn trim_mark(piece: &str) -> &str {
    match piece.strip_prefix(WORD_MARK) {
        Some(rest) => rest,
        None => piece.strip_prefix(' ').unwrap_or(piece),
    }
}

/// Merge `(piece, piece start)` pairs into words.
///
/// `t_end` bounds the last word. Callers should not pass the last piece's own
/// start: the end of the last word is the end of the sentence, and the sentence
/// end is what the latency metric measures against (PLAN §7), so biasing it
/// early flatters every number the project is judged on.
pub fn merge(pieces: &[(String, Duration)], t_end: Duration) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    for (piece, start) in pieces {
        let text = trim_mark(piece);
        if words.is_empty() || starts_word(piece) {
            // A piece that is *only* the marker still opens a word: the pieces
            // after it belong to that new word, not to the one before. Dropping
            // it as empty instead loses the boundary, and "IN NURSING" comes
            // out as "INNURSING" -- which is what this did until the Rust
            // transcript was diffed word-by-word against the Phase 0 PoC's.
            words.push(Word {
                text: text.to_owned(),
                start: *start,
                end: *start,
            });
        } else if let Some(last) = words.last_mut() {
            last.text.push_str(text);
        }
    }
    // A marker that nothing followed leaves an empty word behind.
    words.retain(|w| !w.text.is_empty());
    for i in 1..words.len() {
        words[i - 1].end = words[i].start;
    }
    if let Some(last) = words.last_mut() {
        last.end = t_end.max(last.start);
    }
    words
}

/// Join words back into a line. Both engines are fed one language at a time and
/// neither emits a script that is written without spaces, so this is a join,
/// not a detokeniser.
pub fn text_of(words: &[Word]) -> String {
    words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn pieces(v: &[(&str, u64)]) -> Vec<(String, Duration)> {
        v.iter().map(|(t, m)| ((*t).to_owned(), ms(*m))).collect()
    }

    #[test]
    fn sherpa_pieces_become_words() {
        let w = merge(
            &pieces(&[("▁HE", 100), ("LLO", 200), ("▁WORLD", 400)]),
            ms(600),
        );
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].text, "HELLO");
        assert_eq!(w[1].text, "WORLD");
    }

    #[test]
    fn whisper_pieces_become_words() {
        let w = merge(
            &pieces(&[("Hel", 100), ("lo", 200), (" world", 400)]),
            ms(600),
        );
        assert_eq!(text_of(&w), "Hello world");
    }

    #[test]
    fn punctuation_joins_the_word_before_it() {
        let w = merge(&pieces(&[(" yes", 100), (",", 180), (" no", 300)]), ms(400));
        assert_eq!(text_of(&w), "yes, no");
    }

    #[test]
    fn a_word_ends_where_the_next_one_starts() {
        let w = merge(&pieces(&[("▁ONE", 0), ("▁TWO", 500)]), ms(900));
        assert_eq!(w[0].end, ms(500));
    }

    #[test]
    fn the_last_word_is_closed_by_the_caller_not_by_its_own_start() {
        // The Phase 0 bug: start == end, and every SRT block was zero-length.
        let w = merge(&pieces(&[("▁ONLY", 250)]), ms(900));
        assert_eq!(w[0].start, ms(250));
        assert_eq!(w[0].end, ms(900));
    }

    #[test]
    fn a_t_end_before_the_last_start_does_not_invert_the_span() {
        let w = merge(&pieces(&[("▁LATE", 800)]), ms(700));
        assert!(w[0].end >= w[0].start);
    }

    #[test]
    fn the_first_piece_starts_a_word_even_without_a_marker() {
        // whisper's first token of a segment usually has no leading space.
        let w = merge(&pieces(&[("So", 0), (" then", 100)]), ms(200));
        assert_eq!(text_of(&w), "So then");
    }

    #[test]
    fn a_marker_on_its_own_still_opens_a_word() {
        // sherpa-onnx really does emit the boundary as its own piece. Treating
        // it as "no text, therefore nothing" merged the next word into the
        // previous one -- five times in 82 seconds of read speech.
        let w = merge(
            &pieces(&[
                ("▁IN", 100),
                ("▁", 200),
                ("N", 240),
                ("URS", 300),
                ("ING", 360),
            ]),
            ms(500),
        );
        assert_eq!(text_of(&w), "IN NURSING");
    }

    #[test]
    fn a_trailing_marker_does_not_leave_an_empty_word() {
        let w = merge(&pieces(&[("▁END", 100), ("▁", 300)]), ms(400));
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].end, ms(400));
    }

    #[test]
    fn no_pieces_is_no_words_rather_than_an_empty_word() {
        assert!(merge(&[], ms(100)).is_empty());
    }
}
