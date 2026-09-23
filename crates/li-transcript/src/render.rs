//! One finalised line -> the bytes that go into each file.
//!
//! Nothing here touches the filesystem, so every formatting decision -- the
//! timestamp shape, what happens to a newline inside a sentence, how short a
//! cue is allowed to be -- is a unit test, rather than something you find out
//! about when a player refuses to load the file.

use li_types::TranscriptLine;

/// Shortest subtitle cue that gets written.
///
/// A cue whose start equals its end is invisible in every player, and the
/// Python prototype wrote nothing else: it filled `t_start` and `t_end` with the
/// same value, so every SRT block came out zero-length. The span
/// is real now -- the fast lane fixes it when it hands out the `line_id` -- but
/// a one-word promoted line can still be a couple of hundred milliseconds, so
/// the floor stays.
///
/// It can only push a cue into the one behind it if two lines start less than
/// half a second apart, and a line boundary needs `pause_flush_s` (0.6 s) of
/// silence in front of it, so it cannot.
pub const MIN_CUE_S: f64 = 0.5;

/// WebVTT is only a WebVTT file if it starts with this.
pub const VTT_HEADER: &str = "WEBVTT\n\n";

/// `hh:mm:ss<sep>mmm`. SRT separates the milliseconds with a comma, WebVTT with
/// a full stop, and that single character is the whole difference between the
/// two timing lines.
pub fn stamp(secs: f64, sep: char) -> String {
    let ms = (positive(secs) * 1000.0).round() as u64;
    let (h, rest) = (ms / 3_600_000, ms % 3_600_000);
    let (m, rest) = (rest / 60_000, rest % 60_000);
    let (s, ms) = (rest / 1000, rest % 1000);
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

/// The span a cue is given, after the [`MIN_CUE_S`] floor.
pub fn cue_span(line: &TranscriptLine) -> (f64, f64) {
    let start = positive(line.start_s);
    (start, positive(line.end_s).max(start + MIN_CUE_S))
}

/// One SRT block, blank line included.
pub fn srt(index: usize, line: &TranscriptLine) -> String {
    let (a, b) = cue_span(line);
    format!(
        "{index}\n{} --> {}\n{}\n\n",
        stamp(a, ','),
        stamp(b, ','),
        // SRT has no escaping convention -- players disagree about whether
        // `<i>` is markup -- so the text goes in as it came, minus the newlines
        // that would end the block early.
        one_line(&line.source)
    )
}

/// One WebVTT cue, blank line included. Written after [`VTT_HEADER`].
pub fn vtt(line: &TranscriptLine) -> String {
    let (a, b) = cue_span(line);
    format!(
        "{} --> {}\n{}\n\n",
        stamp(a, '.'),
        stamp(b, '.'),
        escape_vtt(&one_line(&line.source))
    )
}

/// The plain source transcript: one sentence, one line.
pub fn txt(line: &TranscriptLine) -> String {
    format!("{}\n", one_line(&line.source))
}

/// The translation alone, one sentence to a line, for a Chinese-only file.
///
/// A line whose translation never arrived writes nothing, not a blank line: a
/// reader of a Chinese document has no use for a gap where a sentence went.
pub fn txt_translation(translation: &str) -> String {
    let translation = one_line(translation);
    if translation.is_empty() {
        return String::new();
    }
    format!("{translation}\n")
}

/// A source/translation pair for the optional bilingual file.
///
/// `hh:mm:ss` rather than the full cue timing: this file is for reading and for
/// post-editing, and the milliseconds are in the `.srt` and the `.jsonl` for
/// anyone who needs them. A line whose translation never arrived gets its
/// source alone rather than a blank second line.
pub fn bilingual(line: &TranscriptLine, translation: &str) -> String {
    let head = stamp(positive(line.start_s), '.');
    let head = &head[..8];
    let mut out = format!("[{head}] {}\n", one_line(&line.source));
    let translation = one_line(translation);
    if !translation.is_empty() {
        out.push_str(&format!(
            "{:width$}{translation}\n",
            "",
            width = head.len() + 3
        ));
    }
    out.push('\n');
    out
}

/// Newlines, tabs and control characters collapse to single spaces.
///
/// Not cosmetic: a newline inside the text ends an SRT block and a WebVTT cue,
/// so one recognised `\n` would silently truncate the file from that point on
/// for any player that resynchronises on blank lines.
pub fn one_line(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if c.is_control() || c == ' ' {
            space = !out.is_empty();
        } else {
            if space {
                out.push(' ');
            }
            space = false;
            out.push(c);
        }
    }
    out
}

/// WebVTT cue text is markup: `<` opens a span and `&` opens an entity. Doing
/// this also takes care of the spec's other rule -- a cue text line may not
/// contain `-->` -- because the `>` is gone by the time it matters.
pub fn escape_vtt(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

/// NaN and negatives come from arithmetic on an empty span, and would format as
/// a timestamp of `18446744073709551615:00:00`.
fn positive(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 { v } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use li_types::Lane;

    fn line(start_s: f64, end_s: f64, source: &str) -> TranscriptLine {
        TranscriptLine {
            line_id: 1,
            start_s,
            end_s,
            source: source.into(),
            translation: None,
            lane: Lane::Accurate,
            reason: None,
        }
    }

    #[test]
    fn a_timestamp_carries_hours_and_milliseconds() {
        assert_eq!(stamp(0.0, ','), "00:00:00,000");
        assert_eq!(stamp(3723.456, ','), "01:02:03,456");
        assert_eq!(stamp(3723.456, '.'), "01:02:03.456");
    }

    #[test]
    fn a_nonsense_time_formats_as_zero_rather_than_as_five_hundred_million_hours() {
        assert_eq!(stamp(f64::NAN, ','), "00:00:00,000");
        assert_eq!(stamp(-1.0, ','), "00:00:00,000");
    }

    #[test]
    fn a_zero_length_span_is_widened_to_a_visible_cue() {
        // The prototype's bug: both ends were `last_committed_time`, and every
        // player showed nothing at all.
        let (a, b) = cue_span(&line(12.0, 12.0, "hi"));
        assert_eq!((a, b), (12.0, 12.5));
    }

    #[test]
    fn a_real_span_is_left_alone() {
        let (a, b) = cue_span(&line(12.0, 15.25, "hi"));
        assert_eq!((a, b), (12.0, 15.25));
    }

    #[test]
    fn an_srt_block_is_index_timing_text_blank() {
        assert_eq!(
            srt(7, &line(1.5, 4.0, "Hello everybody.")),
            "7\n00:00:01,500 --> 00:00:04,000\nHello everybody.\n\n"
        );
    }

    #[test]
    fn a_vtt_cue_has_no_index_and_a_full_stop() {
        assert_eq!(
            vtt(&line(1.5, 4.0, "Hello everybody.")),
            "00:00:01.500 --> 00:00:04.000\nHello everybody.\n\n"
        );
    }

    #[test]
    fn a_newline_in_the_text_cannot_end_the_cue_early() {
        // What this prevents: everything after the stray newline being read as
        // a new, malformed block.
        let out = srt(1, &line(0.0, 1.0, "one\ntwo\r\nthree"));
        assert!(out.ends_with("one two three\n\n"), "{out}");
    }

    #[test]
    fn vtt_escapes_the_three_characters_that_are_markup() {
        assert_eq!(escape_vtt("a < b & c --> d"), "a &lt; b &amp; c --&gt; d");
    }

    #[test]
    fn runs_of_whitespace_collapse_and_the_ends_are_trimmed() {
        assert_eq!(one_line("  a   b  "), "a b");
        assert_eq!(one_line("   "), "");
    }

    #[test]
    fn a_bilingual_pair_indents_the_translation_under_the_source() {
        assert_eq!(
            bilingual(&line(72.5, 75.0, "Hello everybody."), "您好，各位。"),
            "[00:01:12] Hello everybody.\n           您好，各位。\n\n"
        );
    }

    #[test]
    fn a_translation_line_is_the_text_and_a_newline() {
        assert_eq!(txt_translation("您好，\n各位。"), "您好， 各位。\n");
    }

    #[test]
    fn a_missing_translation_writes_nothing() {
        assert_eq!(txt_translation("  "), "");
    }

    #[test]
    fn a_bilingual_pair_with_no_translation_is_the_source_alone() {
        assert_eq!(
            bilingual(&line(0.0, 1.0, "Hello."), ""),
            "[00:00:00] Hello.\n\n"
        );
    }
}
