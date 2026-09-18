//! The two-line subtitle, redrawn in place.
//!
//! The design wants one source line and one translation line that are *replaced*
//! rather than appended: the fast lane's text appears dimmed within a second,
//! and the accurate lane's replaces it about a second later without the reader
//! losing their place. A scrolling log cannot show that, and it is the one
//! behaviour of the product that has never been looked at by a person -- whether an in-place
//! correction reads as a fix or as flicker is the open question, and this is the cheapest thing that can answer
//! it.
//!
//! Redrawing means erasing what was printed last time, so the bar tracks how
//! many terminal rows it used. Anything else printed in between (a warning on
//! stderr) corrupts one frame; the next event repaints the whole block.

use std::io::{Write, stdout};

use anyhow::Result;
use crossterm::{
    ExecutableCommand, QueueableCommand,
    cursor::{Hide, MoveToPreviousLine, Show},
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{Clear, ClearType},
};
use unicode_width::UnicodeWidthChar;

/// Fallback when the terminal will not say how wide it is.
const FALLBACK_COLUMNS: u16 = 80;

#[derive(Default)]
pub struct Bar {
    /// Rows the last frame occupied, so they can be erased.
    rows: u16,
    source: String,
    /// True while the source line is the fast lane's guess.
    tentative: bool,
    translation: String,
    status: String,
}

impl Bar {
    pub fn open() -> Result<Self> {
        stdout().execute(Hide)?;
        Ok(Self::default())
    }

    pub fn set_status(&mut self, s: impl Into<String>) {
        self.status = s.into();
    }

    /// Fast-lane text, shown dimmed.
    ///
    /// The translation stays. It belongs to an earlier line -- text reaches the
    /// screen 0.8 s after it is spoken and its translation about 4 s after, so
    /// the two rows are only ever the same sentence when the speaker pauses --
    /// but clearing it would blank the row for most of every sentence, and the
    /// translation is what the reader is here for.
    pub fn partial(&mut self, text: &str) {
        self.source = text.to_owned();
        self.tentative = true;
    }

    pub fn source_final(&mut self, text: &str) {
        self.source = text.to_owned();
        self.tentative = false;
    }

    pub fn translation(&mut self, text: &str) {
        self.translation = text.to_owned();
    }

    pub fn draw(&mut self) -> Result<()> {
        let width = crossterm::terminal::size()
            .map(|(w, _)| w.max(20))
            .unwrap_or(FALLBACK_COLUMNS);
        let mut out = stdout().lock();
        if self.rows > 0 {
            out.queue(MoveToPreviousLine(self.rows))?;
        }
        out.queue(Clear(ClearType::FromCursorDown))?;

        let mut rows = 0u16;
        let source = if self.tentative {
            Color::DarkGrey
        } else {
            Color::White
        };
        rows += write_block(&mut out, &self.source, width, source)?;
        rows += write_block(&mut out, &self.translation, width, Color::Cyan)?;
        if !self.status.is_empty() {
            rows += write_block(&mut out, &self.status, width, Color::DarkGrey)?;
        }
        self.rows = rows;
        out.flush()?;
        Ok(())
    }
}

impl Drop for Bar {
    fn drop(&mut self) {
        let _ = stdout().execute(Show);
    }
}

fn write_block(out: &mut impl Write, text: &str, width: u16, colour: Color) -> Result<u16> {
    let lines = wrap(text, width as usize);
    out.queue(SetForegroundColor(colour))?;
    for line in &lines {
        out.queue(Print(line))?;
        out.queue(Print("\n"))?;
    }
    out.queue(ResetColor)?;
    Ok(lines.len() as u16)
}

/// Wrap to `width` display columns, preferring a space.
///
/// By column, not by character: a Han ideograph occupies two of them, so
/// wrapping a Chinese subtitle by `chars()` overflows the terminal by up to
/// double and the bar's row count -- and therefore its erase -- goes wrong.
/// Chinese also has no spaces, so a hard break has to be allowed.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut used = 0usize;
    let mut break_at: Option<(usize, usize)> = None; // (byte index, columns used)

    for c in text.chars() {
        let w = c.width().unwrap_or(0);
        if used + w > width && !line.is_empty() {
            match break_at.filter(|(i, _)| *i > 0) {
                Some((i, _)) => {
                    let rest = line.split_off(i);
                    lines.push(std::mem::take(&mut line).trim_end().to_owned());
                    line = rest.trim_start().to_owned();
                    used = line.chars().filter_map(|c| c.width()).sum();
                }
                None => {
                    lines.push(std::mem::take(&mut line));
                    used = 0;
                }
            }
            break_at = None;
        }
        if c == ' ' {
            break_at = Some((line.len(), used));
        }
        line.push(c);
        used += w;
    }
    lines.push(line);
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_line_is_one_row() {
        assert_eq!(wrap("hello", 20), vec!["hello"]);
        assert_eq!(wrap("", 20), vec![""]);
    }

    #[test]
    fn english_wraps_at_a_space() {
        assert_eq!(
            wrap("the quick brown fox jumps", 12),
            vec!["the quick", "brown fox", "jumps"]
        );
    }

    #[test]
    fn chinese_counts_two_columns_per_character() {
        // The failure this prevents: ten ideographs "fitting" in a 12-column
        // terminal, wrapping themselves, and the bar erasing the wrong rows
        // for the rest of the session.
        let text = "您好各位我是專案經理";
        assert_eq!(text.chars().count(), 10, "ten characters, twenty columns");
        let rows = wrap(text, 12);
        assert_eq!(rows, vec!["您好各位我是", "專案經理"]);
    }

    #[test]
    fn a_word_longer_than_the_terminal_is_broken_rather_than_lost() {
        let rows = wrap("supercalifragilistic", 8);
        assert!(rows.len() >= 3, "{rows:?}");
        assert_eq!(rows.concat(), "supercalifragilistic");
    }
}
