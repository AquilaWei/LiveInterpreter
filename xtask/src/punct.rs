//! `cargo xtask punct` — does the punctuation model find the sentence
//! boundaries the accurate lane finds, and what does asking cost?
//!
//! This is the decision gate for task 1.25 and it deliberately changes no
//! pipeline code. Task 1.24 closed the two acoustic routes to mid-utterance
//! segmentation; what is left is the semantic one, and before any of it is
//! built the hypothesis has to survive a number.
//!
//! ## The ground truth is free
//!
//! whisper punctuates. Over the same audio it already finds the boundaries the
//! endpoint detector misses — 77 of them across the user's five sessions. So
//! the accurate lane's own output is the reference, at no annotation cost and
//! in exactly the domain the feature has to work in. What is scored is:
//!
//! * **truth** — positions in the accurate lane's word stream whose word ends
//!   in `.`, `!` or `?`.
//! * **prediction** — the same, from the punctuation model run over the fast
//!   lane's lower-cased text for that utterance.
//!
//! The two streams are different transcriptions of the same speech, so they are
//! aligned with [`crate::metrics::align`] — the same word-level Levenshtein
//! backtrace the WER numbers use — and a prediction counts as a hit if it lands
//! within [`SLACK`] reference words of a true boundary. The lanes disagree about
//! word identity often enough (5-7 WER points) that demanding an exact index
//! would measure the disagreement rather than the boundary.
//!
//! **Precision matters more than recall.** A missed boundary is today's
//! behaviour. A false one is a fragment on the bar, a short line for
//! `li_stream::merge::decide` to trip its guards on, and a clause handed to
//! NLLB with its subject cut off.

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use li_asr::{DeviceRequest, LaneSpec, punct::OnlinePunct, window::Ring};
use li_audio::file::FileSource;
use li_types::AsrEvent;

use crate::metrics::{self, Op};

const SAMPLE_RATE: f64 = 16_000.0;

/// How far from a true boundary a prediction may land and still count.
///
/// One word. The two lanes are different transcriptions, so their word indices
/// drift; and in practice a boundary placed one word early or late still cuts
/// at a clause edge, which is what the feature is for. Two would start counting
/// mid-clause guesses as hits.
const SLACK: usize = 1;

const MARKS: [char; 3] = ['.', '!', '?'];

pub struct Args {
    wav: PathBuf,
    /// Score against text instead of audio: see [`go_text`].
    text: Option<PathBuf>,
    fast_model: Option<PathBuf>,
    accurate_model: Option<PathBuf>,
    punct_model: Option<PathBuf>,
    device: DeviceRequest,
    threads: usize,
    dump: bool,
}

pub fn parse(mut it: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args {
        wav: PathBuf::new(),
        text: None,
        fast_model: None,
        accurate_model: None,
        punct_model: None,
        device: DeviceRequest::Auto,
        threads: 1,
        dump: false,
    };
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("a flag is missing its value"))
        };
        match flag.as_str() {
            "--wav" => a.wav = val()?.into(),
            "--text" => a.text = Some(val()?.into()),
            "--fast-model" => a.fast_model = Some(val()?.into()),
            "--accurate-model" => a.accurate_model = Some(val()?.into()),
            "--punct-model" => a.punct_model = Some(val()?.into()),
            "--device" => a.device = val()?.parse()?,
            "--threads" => a.threads = val()?.parse()?,
            "--dump" => a.dump = true,
            other => bail!("unknown flag {other}"),
        }
    }
    if a.wav.as_os_str().is_empty() && a.text.is_none() {
        bail!(
            "usage: cargo xtask punct (--wav <clip.wav> | --text <file>) [--fast-model P] \
             [--accurate-model P] [--punct-model P] [--device D] [--threads N] [--dump]"
        );
    }
    Ok(a)
}

fn cache() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/liveinterpreter/models")
}

/// One utterance, seen by both lanes.
struct Utterance {
    t_start: Duration,
    t_end: Duration,
    /// The fast lane's own words, upper case and unpunctuated.
    fast: String,
    /// The same audio through the punctuation model.
    punctuated: String,
    /// The accurate lane's answer: the reference.
    accurate: String,
    /// What `restore` cost, and over how many words.
    ms: f64,
    words: usize,
}

pub fn run(args: Args) -> Result<()> {
    if args.text.is_some() {
        return go_text(&args);
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(go(args))
}

/// Score the model against punctuated text, with no ASR in the way.
///
/// `--wav` measures two things at once and cannot separate them. On a clip
/// sped up to 1.5x the fast lane mishears badly enough that the punctuator is
/// working on different words than the reference was written from -- so a
/// low score there is evidence about the recogniser, not about boundaries.
///
/// This asks the narrow question instead: **given text that already has
/// sentence ends in it, stripped of its marks and shouted, does the model put
/// them back?** Input and reference are then the same words by construction.
/// It accepts one line per line, or `.jsonl` with a `source` field -- which is
/// what the program's own transcripts are, so the target domain is available
/// without recording anything.
fn go_text(args: &Args) -> Result<()> {
    let path = args.text.as_ref().expect("checked by the caller");
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let lines: Vec<String> = raw
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            if l.is_empty() {
                return None;
            }
            if l.starts_with('{') {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                // A `lane` of "fast" carries no punctuation to recover.
                if v.get("lane").and_then(|l| l.as_str()) == Some("fast") {
                    return None;
                }
                return v.get("source").and_then(|s| s.as_str()).map(str::to_owned);
            }
            Some(l.to_owned())
        })
        .collect();

    let dir = args
        .punct_model
        .clone()
        .unwrap_or_else(li_asr::punct::default_model_dir);
    let mut punct = OnlinePunct::open(&dir, args.threads)?;

    let mut utts = Vec::new();
    for line in &lines {
        // What the fast lane would have handed over: the same words, shouted,
        // with every mark taken out.
        let stripped = line
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '\'')
                    .flat_map(char::to_uppercase)
                    .collect::<String>()
            })
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if stripped.split_whitespace().count() < 2 {
            continue;
        }
        let started = Instant::now();
        let punctuated = punct.restore(&stripped)?;
        let ms = started.elapsed().as_secs_f64() * 1e3;
        utts.push(Utterance {
            t_start: Duration::ZERO,
            t_end: Duration::ZERO,
            words: stripped.split_whitespace().count(),
            fast: stripped,
            punctuated,
            accurate: line.clone(),
            ms,
        });
    }
    report(args, &utts, punct.mismatches());
    Ok(())
}

async fn go(args: Args) -> Result<()> {
    let mut spec = LaneSpec::new(
        "sherpa",
        args.fast_model
            .clone()
            .unwrap_or_else(|| cache().join("sherpa-onnx-streaming-zipformer-en-2023-06-21")),
    );
    spec.device = DeviceRequest::Cpu;
    spec.threads = 2;
    let mut fast = li_asr::build(&spec)?;

    let mut acc_spec = LaneSpec::new(
        "whispercpp",
        args.accurate_model
            .clone()
            .unwrap_or_else(|| cache().join("ggml/ggml-small.en-q5_1.bin")),
    );
    acc_spec.device = args.device;
    acc_spec.threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).max(1))
        .unwrap_or(4);
    let mut accurate = li_asr::build(&acc_spec)?;

    let dir = args
        .punct_model
        .clone()
        .unwrap_or_else(li_asr::punct::default_model_dir);
    let mut punct = OnlinePunct::open(&dir, args.threads)?;

    // Not realtime: nothing here is a latency measurement of the pipeline, and
    // the accurate lane is the slow part either way.
    let mut rx = FileSource::new(&args.wav)
        .with_agc(true)
        .with_realtime(false)
        .open()?;

    let mut samples: u64 = 0;
    let mut ring = Ring::default();
    let mut out: Vec<Utterance> = Vec::new();

    while let Some(frame) = rx.recv().await {
        let t_origin = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
        samples += frame.pcm.len() as u64;
        let now = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
        ring.push(&frame.pcm);

        for ev in fast.feed(&frame.pcm, t_origin).await? {
            let AsrEvent::Final {
                words,
                t_start,
                t_end,
                ..
            } = ev
            else {
                continue;
            };
            if words.is_empty() {
                continue;
            }
            let started = Instant::now();
            let punctuated = punct.restore_words(&words)?;
            let ms = started.elapsed().as_secs_f64() * 1e3;

            // The same window `li_core::Engine` gives whisper, by construction.
            let (pcm, a) = ring.cut(t_start, t_end.min(now));
            let evs = accurate.feed(&pcm, a).await?;
            let acc = evs
                .into_iter()
                .find_map(|e| match e {
                    AsrEvent::Partial { words, .. } => Some(words),
                    _ => None,
                })
                .unwrap_or_default();

            out.push(Utterance {
                t_start,
                t_end,
                words: words.len(),
                fast: li_asr::words::text_of(&words),
                punctuated: li_asr::words::text_of(&punctuated),
                accurate: li_asr::words::text_of(&acc),
                ms,
            });
        }
    }

    report(&args, &out, punct.mismatches());
    Ok(())
}

/// Token stream plus the token indices that end a sentence.
///
/// Built one whitespace word at a time so the mapping is exact. Going through
/// [`metrics::tokens`] on the whole string and hoping the counts line up does
/// not work: it spells numbers out and splits on hyphens, so "twenty-five"
/// becomes two tokens and every index after it is wrong. That silently threw
/// away 8 of one clip's 27 utterances before this was written word-wise.
///
/// The mark is looked for before tokenising, because tokenising removes it.
/// A boundary is recorded against the *last* token the word produced, so
/// "twenty-five." lands on "five".
fn boundaries(text: &str) -> (Vec<String>, Vec<usize>, usize) {
    let mut toks = Vec::new();
    let mut ends = Vec::new();
    let mut last_word_start = 0usize;
    for raw in text.split_whitespace() {
        let mark = raw
            .trim_end_matches(['"', '\'', ')', ']', '\u{201d}'])
            .ends_with(MARKS);
        let piece = metrics::tokens(raw, false);
        if piece.is_empty() {
            continue;
        }
        last_word_start = toks.len();
        toks.extend(piece);
        if mark {
            ends.push(toks.len() - 1);
        }
    }
    (toks, ends, last_word_start)
}

/// Hits, predictions and truths for one class of boundary.
#[derive(Default)]
struct Score {
    hits: usize,
    predicted: usize,
    truth: usize,
}

impl Score {
    /// `keep` selects which true boundaries this class counts; a prediction is
    /// only counted against the class whose truths it could have matched, so
    /// the final-word boundary does not inflate the inner precision.
    fn score(&mut self, mapped: &[(usize, bool)], real: &[usize], keep: impl Fn(usize) -> bool) {
        let truths: Vec<usize> = real.iter().copied().filter(|t| keep(*t)).collect();
        self.truth += truths.len();
        let mut taken = vec![false; truths.len()];
        for (m, _) in mapped {
            let hit = truths
                .iter()
                .enumerate()
                .filter(|(k, _)| !taken[*k])
                .find(|(_, t)| t.abs_diff(*m) <= SLACK)
                .map(|(k, _)| k);
            // Only predictions that could plausibly belong to this class count
            // against it: one that matched nothing here may have matched the
            // other class, and charging it to both would double-count.
            match hit {
                Some(k) => {
                    taken[k] = true;
                    self.hits += 1;
                    self.predicted += 1;
                }
                None => {
                    if real.iter().all(|t| t.abs_diff(*m) > SLACK) {
                        self.predicted += 1;
                    }
                }
            }
        }
    }

    fn precision(&self) -> f64 {
        if self.predicted == 0 {
            f64::NAN
        } else {
            self.hits as f64 / self.predicted as f64
        }
    }

    fn recall(&self) -> f64 {
        if self.truth == 0 {
            f64::NAN
        } else {
            self.hits as f64 / self.truth as f64
        }
    }
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[i]
}

fn report(args: &Args, utts: &[Utterance], mismatches: u64) {
    let name = args
        .text
        .as_ref()
        .unwrap_or(&args.wav)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    println!("# xtask punct — {name}\n");

    let mut all = Score::default();
    // The boundary that lands on an utterance's last word is the endpoint doing
    // its job -- sherpa already cut there. What this feature exists to find is
    // the *other* kind, and scoring both together buries it: on a clip that
    // segments cleanly nearly every true boundary is a final one, so recall
    // reads low no matter how good the model is inside a line.
    let mut inner = Score::default();
    let mut ms: Vec<f64> = Vec::new();
    let mut per_word: Vec<f64> = Vec::new();

    for u in utts {
        ms.push(u.ms);
        if u.words > 0 {
            per_word.push(u.ms / u.words as f64);
        }
        let (hyp, pred, hyp_last) = boundaries(&u.punctuated);
        let (refr, real, ref_last) = boundaries(&u.accurate);
        if hyp.is_empty() || refr.is_empty() {
            continue;
        }

        // Map every hypothesis position onto a reference position, then score
        // in reference space so both sides are counted on one axis.
        let ops = metrics::align(&refr, &hyp);
        let mut hyp_to_ref = vec![usize::MAX; hyp.len()];
        let mut j = 0usize;
        for (op, ri) in ops {
            match op {
                Op::Equal | Op::Sub => {
                    if j < hyp.len() {
                        hyp_to_ref[j] = ri;
                    }
                    j += 1;
                }
                Op::Insert => {
                    if j < hyp.len() {
                        hyp_to_ref[j] = ri.min(refr.len().saturating_sub(1));
                    }
                    j += 1;
                }
                Op::Delete => {}
            }
        }

        let mapped: Vec<(usize, bool)> = pred
            .iter()
            .filter_map(|&p| hyp_to_ref.get(p).copied().map(|r| (r, p >= hyp_last)))
            .filter(|(r, _)| *r != usize::MAX)
            .collect();

        all.score(&mapped, &real, |_| true);
        inner.score(&mapped, &real, |t| t < ref_last);
    }

    println!("| what | all boundaries | **inside a line** |");
    println!("|---|---|---|");
    println!("| utterances | {} | |", utts.len());
    println!(
        "| true sentence ends | {} | **{}** |",
        all.truth, inner.truth
    );
    println!("| predicted | {} | {} |", all.predicted, inner.predicted);
    println!("| hits (±{SLACK} words) | {} | {} |", all.hits, inner.hits);
    println!(
        "| **precision** | {:.2} | **{:.2}** |",
        all.precision(),
        inner.precision()
    );
    println!(
        "| **recall** | {:.2} | **{:.2}** |",
        all.recall(),
        inner.recall()
    );
    println!("| word-count mismatches | {mismatches} | |");

    // A file with no marks in it has nothing to be right or wrong about, and
    // printing 0.00 precision against an empty truth reads like a result. The
    // WER references in `testdata/` are exactly this shape -- upper case, no
    // punctuation, one line -- so the trap is one tab-completion away.
    if all.truth == 0 {
        println!(
            "\n**Nothing to score.** The reference carries no `.`, `!` or `?`, so every \
             prediction is counted wrong by default. `testdata/*.en.txt` are WER \
             references and are unpunctuated on purpose; score against a `.jsonl` \
             transcript or a prose reference instead."
        );
    }

    ms.sort_by(f64::total_cmp);
    per_word.sort_by(f64::total_cmp);
    println!("\n| restore cost | n | p50 | p90 | max |");
    println!("|---|---|---|---|---|");
    println!(
        "| ms per call | {} | {:.1} | {:.1} | {:.1} |",
        ms.len(),
        pct(&ms, 0.5),
        pct(&ms, 0.9),
        pct(&ms, 1.0)
    );
    println!(
        "| ms per word | {} | {:.2} | {:.2} | {:.2} |",
        per_word.len(),
        pct(&per_word, 0.5),
        pct(&per_word, 0.9),
        pct(&per_word, 1.0)
    );

    if args.dump {
        println!("\n## Utterances\n");
        for u in utts {
            println!(
                "**{:.1}–{:.1}s** ({} words, {:.1} ms)\n- fast: {}\n- punct: {}\n- acc:  {}\n",
                u.t_start.as_secs_f64(),
                u.t_end.as_secs_f64(),
                u.words,
                u.ms,
                u.fast,
                u.punctuated,
                u.accurate
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sentence_end_is_found_through_a_closing_quote() {
        let (toks, ends, last) = boundaries("i am fine. \"thank you.\" and you");
        assert_eq!(toks.len(), 7);
        // "fine." at 2, and `you."` at 4 -- the quote must not hide the stop.
        assert_eq!(ends, vec![2, 4]);
        assert_eq!(last, 6, "the last word starts at the last token here");
    }

    #[test]
    fn a_comma_is_not_a_sentence_end() {
        let (_, ends, _) = boundaries("well, i think so");
        assert!(ends.is_empty(), "{ends:?}");
    }

    #[test]
    fn a_word_the_tokeniser_splits_still_lines_up() {
        // `metrics::tokens` spells numbers out, so "25" becomes two tokens and
        // a whole-string tokenisation would put every later index one out.
        // Word-wise, the boundary lands on the last token the word produced.
        let (toks, ends, _) = boundaries("we have 25 minutes. right");
        assert_eq!(toks, ["we", "have", "twenty", "five", "minutes", "right"]);
        assert_eq!(ends, vec![4], "the stop belongs to `minutes`");
    }

    #[test]
    fn the_inner_class_ignores_a_boundary_on_the_last_word() {
        // One prediction, one truth, both on the final word: `all` scores it,
        // `inner` sees neither, because the endpoint detector already cut there.
        let mut all = Score::default();
        let mut inner = Score::default();
        let mapped = [(5usize, true)];
        let real = [5usize];
        let ref_last = 5usize;
        all.score(&mapped, &real, |_| true);
        inner.score(&mapped, &real, |t| t < ref_last);
        assert_eq!((all.hits, all.truth, all.predicted), (1, 1, 1));
        assert_eq!((inner.hits, inner.truth, inner.predicted), (0, 0, 0));
    }

    #[test]
    fn a_prediction_nowhere_near_a_truth_costs_precision() {
        let mut s = Score::default();
        s.score(&[(2, false)], &[9], |_| true);
        assert_eq!((s.hits, s.predicted, s.truth), (0, 1, 1));
        assert!(s.precision().abs() < f64::EPSILON);
    }
}
