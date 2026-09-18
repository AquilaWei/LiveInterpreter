//! Subtitle-oriented ASR metrics.
//!
//! A Rust port of the Python prototype's `metrics.py`, which is the definition
//! of record for gates G2a and G2b; the two must agree, so the tokeniser and
//! the alignment are ported behaviour-for-behaviour rather than rewritten.
//!
//! Plain WER is a poor proxy for this product. It charges the model equally for
//! dropping "um" -- which makes a subtitle *better* -- and for turning "tool
//! training" into "teal training", which becomes a wrong Chinese sentence. And
//! it hides the failure mode that actually ruins the experience: a whole clause
//! going missing, which looks like a handful of deletions mixed in with
//! harmless ones.
//!
//! So two numbers are reported instead:
//!
//! * `content_wer` -- WER after removing from BOTH sides the words that do not
//!   survive into a subtitle anyway (fillers, backchannels, truncated
//!   half-words), and after normalising the two ways a correct transcription
//!   can still differ in text: digits vs number words, and British vs American
//!   spelling.
//! * `drop_rate` -- reference words lost in runs of >= `min_run` consecutive
//!   deletions, i.e. how much of the talk vanished in clause-sized holes.
//!
//! Removal is symmetric, so it cannot flatter the hypothesis: a word the
//! reference does not have cannot be deleted from it.

/// Overwhelmingly discourse in meeting speech. Deliberately excludes "right",
/// "well", "so" and "yes"/"no", which carry content often enough to matter.
const FILLER: &[&str] = &[
    "um", "uh", "er", "erm", "ah", "eh", "huh", "hmm", "mm", "mhm", "mmm", "uhhuh", "mmhmm",
    "yeah", "yep", "yup", "okay", "ok", "kay",
];

/// whisper writes American; AMI writes British. Both are correct.
const BRITISH: &[(&str, &str)] = &[
    ("favourite", "favorite"),
    ("colour", "color"),
    ("colours", "colors"),
    ("behaviour", "behavior"),
    ("centre", "center"),
    ("metre", "meter"),
    ("realise", "realize"),
    ("realised", "realized"),
    ("organise", "organize"),
    ("organised", "organized"),
    ("recognise", "recognize"),
    ("summarise", "summarize"),
    ("summarisation", "summarization"),
    ("analyse", "analyze"),
    ("programme", "program"),
    ("practise", "practice"),
    ("grey", "gray"),
    ("travelling", "traveling"),
];

const ONES: &[&str] = &[
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: &[&str] = &[
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Digits -> the words a transcriber would have written.
fn spell(n: u64, out: &mut Vec<String>) {
    match n {
        0..=19 => out.push(ONES[n as usize].into()),
        20..=99 => {
            out.push(TENS[(n / 10) as usize].into());
            if n % 10 != 0 {
                spell(n % 10, out);
            }
        }
        100..=999 => {
            spell(n / 100, out);
            out.push("hundred".into());
            if n % 100 != 0 {
                spell(n % 100, out);
            }
        }
        _ => {
            spell(n / 1000, out);
            out.push("thousand".into());
            if n % 1000 != 0 {
                spell(n % 1000, out);
            }
        }
    }
}

pub fn tokens(text: &str, content_only: bool) -> Vec<String> {
    let flattened: String = text
        .chars()
        .map(|c| {
            // `_` counts as a word character, as it does in the Python port's
            // `\w`: AMI writes spelled-out letters as "I_D", and splitting that
            // into "i" and "d" would leave a stray "i" in the reference.
            if c.is_alphanumeric() || c == '_' || c == '\'' || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    let mut out = Vec::new();
    for raw in flattened.split_whitespace() {
        let w = raw.to_lowercase();
        if w.len() <= 6 && w.chars().all(|c| c.is_ascii_digit()) {
            spell(w.parse().unwrap_or(0), &mut out);
            continue;
        }
        let w = BRITISH
            .iter()
            .find(|(b, _)| *b == w)
            .map_or(w, |(_, a)| (*a).to_owned());
        if content_only {
            // "_" is how AMI writes spelled-out letters; a bare letter is a
            // half-word the speaker cut off. Neither belongs in a subtitle.
            if FILLER.contains(&w.as_str()) || w.contains('_') {
                continue;
            }
            if w.chars().count() == 1 && w != "a" && w != "i" {
                continue;
            }
        }
        out.push(w);
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    Equal,
    Sub,
    Delete,
    Insert,
}

/// Levenshtein backtrace over words, as `(op, reference index)` per step.
pub(crate) fn align(r: &[String], h: &[String]) -> Vec<(Op, usize)> {
    let (m, n) = (r.len(), h.len());
    let mut d = vec![0usize; (m + 1) * (n + 1)];
    let at = |i: usize, j: usize| i * (n + 1) + j;
    for i in 0..=m {
        d[at(i, 0)] = i;
    }
    for j in 0..=n {
        d[at(0, j)] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let sub = d[at(i - 1, j - 1)] + usize::from(r[i - 1] != h[j - 1]);
            d[at(i, j)] = sub.min(d[at(i - 1, j)] + 1).min(d[at(i, j - 1)] + 1);
        }
    }

    let (mut i, mut j) = (m, n);
    let mut ops = Vec::new();
    while i > 0 || j > 0 {
        // Order matters only for which of several equal-cost paths is reported;
        // diagonal first keeps a substitution from being split into a delete
        // plus an insert, which would inflate the drop rate.
        if i > 0 && j > 0 && d[at(i, j)] == d[at(i - 1, j - 1)] + usize::from(r[i - 1] != h[j - 1])
        {
            ops.push((
                if r[i - 1] == h[j - 1] {
                    Op::Equal
                } else {
                    Op::Sub
                },
                i - 1,
            ));
            i -= 1;
            j -= 1;
        } else if i > 0 && d[at(i, j)] == d[at(i - 1, j)] + 1 {
            ops.push((Op::Delete, i - 1));
            i -= 1;
        } else {
            ops.push((Op::Insert, i));
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

#[derive(Debug, Clone)]
pub struct Metrics {
    /// Plain WER, for continuity with the earliest reports.
    pub wer: f64,
    /// Gate G2a.
    pub content_wer: f64,
    /// Gate G2b.
    pub drop_rate: f64,
    /// The holes themselves, for eyeballing.
    pub drops: Vec<String>,
    pub ref_words: usize,
}

fn wer_of(r: &[String], h: &[String]) -> f64 {
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 100.0 };
    }
    let errors = align(r, h)
        .iter()
        .filter(|(op, _)| *op != Op::Equal)
        .count();
    errors as f64 / r.len() as f64 * 100.0
}

pub fn measure(reference: &str, hypothesis: &str, min_run: usize) -> Metrics {
    let (r, h) = (tokens(reference, true), tokens(hypothesis, true));
    let ops = align(&r, &h);

    let mut drops: Vec<String> = Vec::new();
    let mut run: Vec<usize> = Vec::new();
    for (op, idx) in ops.iter().chain(std::iter::once(&(Op::Equal, 0))) {
        if *op == Op::Delete {
            run.push(*idx);
        } else {
            if run.len() >= min_run {
                drops.push(
                    run.iter()
                        .map(|i| r[*i].as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
            run.clear();
        }
    }
    let dropped: usize = drops.iter().map(|d| d.split_whitespace().count()).sum();

    Metrics {
        wer: wer_of(&tokens(reference, false), &tokens(hypothesis, false)),
        content_wer: wer_of(&r, &h),
        drop_rate: if r.is_empty() {
            0.0
        } else {
            dropped as f64 / r.len() as f64 * 100.0
        },
        drops,
        ref_words: r.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_become_the_words_a_transcriber_would_have_written() {
        assert_eq!(tokens("25 items", false), ["twenty", "five", "items"]);
        assert_eq!(
            tokens("1500", false),
            ["one", "thousand", "five", "hundred"]
        );
        assert_eq!(tokens("7", false), ["seven"]);
    }

    #[test]
    fn british_and_american_spellings_are_the_same_transcription() {
        let m = measure(
            "we should summarise the colour",
            "We should summarize the color.",
            5,
        );
        assert_eq!(m.content_wer, 0.0);
    }

    #[test]
    fn dropping_a_filler_is_not_an_error_but_a_word_is() {
        let m = measure("um so we have the agenda", "So we have the agenda.", 5);
        assert_eq!(m.content_wer, 0.0);
        assert!(m.wer > 0.0, "plain WER still counts it: {}", m.wer);

        let m = measure("so we have the agenda", "So we have the plan.", 5);
        assert!((m.content_wer - 20.0).abs() < 1e-9, "{}", m.content_wer);
    }

    #[test]
    fn a_clause_sized_hole_is_a_drop_and_scattered_deletions_are_not() {
        let m = measure(
            "this is our first meeting surprisingly enough this is our agenda",
            "this is our agenda",
            5,
        );
        assert_eq!(m.drops.len(), 1);
        // Which seven words are named depends on which of two equal-cost
        // alignments the backtrace takes -- "this is our" appears twice in the
        // reference. The size of the hole does not.
        assert_eq!(m.drops[0].split_whitespace().count(), 7, "{}", m.drops[0]);
        assert!(
            (m.drop_rate - 7.0 / 11.0 * 100.0).abs() < 1e-9,
            "{}",
            m.drop_rate
        );

        // Four scattered single-word deletions: bad WER, but no hole.
        let m = measure(
            "one two three four five six seven eight",
            "one three five seven",
            5,
        );
        assert_eq!(m.drop_rate, 0.0);
        assert!(m.content_wer > 0.0);
    }

    #[test]
    fn a_repetition_loop_shows_up_as_insertions_not_as_a_drop() {
        let m = measure(
            "and then when you go on the menu",
            "and then when you go on the menu and then when you go on the menu",
            5,
        );
        assert_eq!(m.drop_rate, 0.0);
        assert!((m.content_wer - 100.0).abs() < 1e-9, "{}", m.content_wer);
    }

    #[test]
    fn an_empty_hypothesis_loses_everything() {
        let m = measure("one two three four five six", "", 5);
        assert_eq!(m.drop_rate, 100.0);
        assert_eq!(m.content_wer, 100.0);
    }
}

// ---------------------------------------------------------------------------
// chrF -- the MT metric
// ---------------------------------------------------------------------------

/// Character n-gram F-score, as sacrebleu computes it with default settings:
/// `char_order = 6`, no word n-grams, `beta = 2`, whitespace removed, no eps
/// smoothing. Returned on sacrebleu's 0-100 scale.
///
/// Character n-grams, not words, is what makes this the right family of metric
/// for Chinese: it needs no segmenter, and it gives partial credit for a
/// translation that gets the phrase nearly right -- which is most of what
/// distinguishes a readable subtitle from a bad one.
///
/// Ported rather than reimplemented: `tests` checks it against sacrebleu 2.6.0
/// on fixed pairs, because a metric that disagrees with the published one is
/// worse than no metric.
pub fn chrf(reference: &str, hypothesis: &str) -> f64 {
    const ORDER: usize = 6;
    const BETA2: f64 = 4.0;

    let r: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();
    let h: Vec<char> = hypothesis.chars().filter(|c| !c.is_whitespace()).collect();

    let (mut sum_prec, mut sum_rec, mut orders) = (0.0, 0.0, 0usize);
    for n in 1..=ORDER {
        let (rc, hc) = (ngrams(&r, n), ngrams(&h, n));
        let (n_ref, n_hyp) = (count(&rc), count(&hc));
        if n_ref == 0 || n_hyp == 0 {
            continue;
        }
        let matched: usize = hc
            .iter()
            .map(|(g, c)| (*c).min(rc.get(g).copied().unwrap_or(0)))
            .sum();
        sum_prec += matched as f64 / n_hyp as f64;
        sum_rec += matched as f64 / n_ref as f64;
        orders += 1;
    }
    if orders == 0 || sum_prec + sum_rec == 0.0 {
        return 0.0;
    }
    let (prec, rec) = (sum_prec / orders as f64, sum_rec / orders as f64);
    100.0 * (1.0 + BETA2) * prec * rec / (BETA2 * prec + rec)
}

fn ngrams(cs: &[char], n: usize) -> std::collections::HashMap<&[char], usize> {
    let mut m = std::collections::HashMap::new();
    if cs.len() >= n {
        for w in cs.windows(n) {
            *m.entry(w).or_insert(0) += 1;
        }
    }
    m
}

fn count(m: &std::collections::HashMap<&[char], usize>) -> usize {
    m.values().sum()
}

#[cfg(test)]
mod chrf_tests {
    use super::chrf;

    /// Against sacrebleu 2.6.0 itself, on pairs that cover the shapes this is
    /// asked about: a near-miss, a paraphrase, no overlap at all, an identical
    /// string, an empty reference, a hypothesis far longer than the reference,
    /// two full documents, and Latin text where whitespace has to be ignored.
    #[test]
    fn it_agrees_with_sacrebleu() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/chrf_reference.json");
        let raw = std::fs::read_to_string(path).expect("chrf fixture");
        let doc: serde_json::Value = serde_json::from_str(&raw).expect("valid fixture");
        let pairs = doc["pairs"].as_array().expect("pairs");
        assert!(pairs.len() >= 8);
        for p in pairs {
            let (r, h) = (p["ref"].as_str().unwrap(), p["hyp"].as_str().unwrap());
            let want = p["chrf"].as_f64().unwrap();
            let got = chrf(r, h);
            assert!(
                (got - want).abs() < 1e-4,
                "chrF differs on {:?} / {:?}: sacrebleu {want}, ours {got}",
                r.chars().take(24).collect::<String>(),
                h.chars().take(24).collect::<String>(),
            );
        }
    }

    #[test]
    fn an_empty_side_scores_zero_rather_than_dividing_by_zero() {
        assert_eq!(chrf("", ""), 0.0);
        assert_eq!(chrf("一模一樣", ""), 0.0);
    }
}
