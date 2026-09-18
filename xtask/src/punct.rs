//! `cargo xtask punct` — does the punctuation model find the sentence
//! boundaries the accurate lane finds, and what does asking cost?
//!
//! This is the decision gate for punctuation restoration and it deliberately changes no
//! pipeline code. Earlier measurements closed the two acoustic routes to mid-utterance
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
//!
//! ## E2 — and how much earlier
//!
//! Being right is necessary and not sufficient. A boundary the punctuator only
//! finds at the same moment the endpoint detector fires buys nothing, and the
//! plan's own gate says so: **median `t_close − t_decide` below 1.0 s and
//! neither the early translation nor stage S2 is worth building.**
//!
//! Until now that number had never been measured. The 2.88 s median from the
//! user's transcripts is whisper's hindsight — it saw the whole utterance
//! before deciding, which a streaming punctuator cannot. So [`Watch`] runs the
//! punctuator over **every fast-lane partial** and records, per boundary:
//!
//! * `t_close − t_stable` — **E2 proper**: the wait the line would have been
//!   spared. `t_close` is the audio position at which the fast lane's `Final`
//!   actually arrived, so this is measured against what happens today, trailing
//!   silence and all.
//! * `t_close − t_first` — the same with no stability rule at all. The gap
//!   between the two is what the rule costs.
//! * `t_stable − t_boundary` — how long after the word itself the decision
//!   could be taken. If this routinely exceeds `endpoint_silence_s` (0.6 s) the
//!   endpoint detector would have got there first and the feature only rescues
//!   runaway lines.
//! * **flaps** — how many times the mark appeared and vanished again before the
//!   line closed. This is the data the stability thresholds should be read
//!   from, rather than guessed at by analogy with `SPECULATE_AFTER`.
//!
//! Every time here is **audio time**, never the wall clock: the question is
//! when the information arrived, and a clock would make the answer depend on
//! how fast this machine ran the file.
//!
//! The population is the fast lane's *own* inner predictions at close time —
//! not whisper's boundaries — because those are exactly the cuts a semantic
//! segmenter would act on. Whether they deserve to be trusted is the precision
//! table's question; this one asks only how much sooner they were there.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use li_asr::{DeviceRequest, LaneSpec, punct::OnlinePunct, window::Ring};
use li_audio::file::FileSource;
use li_types::{AsrEvent, Word};

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

/// The shared model cache, by the one rule that decides it.
///
/// Spelled out by hand here until the Flatpak work: three copies of
/// `$HOME/.cache/liveinterpreter/models`, which is the exact duplication
/// `li_types::paths` was written to remove (see its module docs). They went
/// wrong together when the flatpak work taught that rule about
/// `XDG_CACHE_HOME`, and a harness that looks somewhere the app does not is a
/// harness that measures nothing.
fn cache() -> PathBuf {
    li_types::paths::model_cache()
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
    // No partials on this path, so no E2: `--text` has no audio timeline to
    // measure one against.
    report(args, &utts, punct.mismatches(), &Watch::default());
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
    let mut watch = Watch::default();

    while let Some(frame) = rx.recv().await {
        let t_origin = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
        samples += frame.pcm.len() as u64;
        let now = Duration::from_secs_f64(samples as f64 / SAMPLE_RATE);
        ring.push(&frame.pcm);

        for ev in fast.feed(&frame.pcm, t_origin).await? {
            // The fast lane's partials carry no word times -- the one thing
            // this tool needs that the pipeline does not hand it -- but they do
            // carry the text, and `now` says where in the audio it arrived.
            // That is all E2 asks for.
            let AsrEvent::Final {
                words,
                t_start,
                t_end,
                ..
            } = ev
            else {
                let AsrEvent::Partial { text, .. } = ev else {
                    unreachable!("there are two variants")
                };
                watch.observe(&text, now, &mut punct)?;
                continue;
            };
            if words.is_empty() {
                continue;
            }
            let started = Instant::now();
            let punctuated = punct.restore_words(&words)?;
            let ms = started.elapsed().as_secs_f64() * 1e3;
            watch.close(&punctuated, now);

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

    report(&args, &out, punct.mismatches(), &watch);
    Ok(())
}

/// One boundary the punctuator saw inside a line that was still open.
struct Timing {
    /// Audio time of the first partial whose punctuation showed it.
    first: Duration,
    /// Audio time at which it satisfied the stability rule, if it ever did.
    stable: Option<Duration>,
    /// How many times it went away again after having been seen.
    flaps: usize,
    /// Whether the run just processed still showed it, so a disappearance is
    /// counted once rather than once per partial.
    present: bool,
    /// Consecutive runs showing it.
    streak: usize,
}

/// What one closed line contributed to E2. One row per inner boundary.
struct Early {
    /// `t_close - t_stable`: the wait the line would have been spared.
    saved: Option<f64>,
    /// The same from first sight, with no stability rule at all.
    saved_raw: Option<f64>,
    /// `t_stable - t_boundary`: how long after the word itself the decision
    /// could be taken.
    lag: Option<f64>,
    flaps: usize,
}

/// Runs the punctuator over the partials of the line that is currently open.
///
/// Deliberately not part of the scoring above: that asks whether a boundary is
/// real, this asks when it could have been known. Keeping them apart means a
/// bad answer to one does not quietly contaminate the other.
#[derive(Default)]
struct Watch {
    seen: BTreeMap<usize, Timing>,
    rows: Vec<Early>,
    ms: Vec<f64>,
    /// Partials the model handed back a different number of words for. Their
    /// indices mean nothing, so they are skipped rather than guessed at --
    /// the same refusal `restore_words` makes for the same reason.
    skipped: u64,
    calls: u64,
}

impl Watch {
    /// Punctuate one partial and fold it into what is already known.
    ///
    /// `now` is audio time. See the module docs on why it must not be a clock.
    fn observe(&mut self, text: &str, now: Duration, punct: &mut OnlinePunct) -> Result<()> {
        let n = text.split_whitespace().count();
        if n < 2 {
            return Ok(());
        }
        let started = Instant::now();
        let out = punct.restore(text)?;
        self.ms.push(started.elapsed().as_secs_f64() * 1e3);
        self.calls += 1;

        let pieces: Vec<&str> = out.split_whitespace().collect();
        if pieces.len() != n {
            self.skipped += 1;
            return Ok(());
        }
        let ends: Vec<usize> = pieces
            .iter()
            .enumerate()
            .filter(|(_, p)| ends_sentence(p))
            .map(|(i, _)| i)
            .collect();
        self.fold(&ends, n, now);
        Ok(())
    }

    /// The stability rule, with the model taken out of the way.
    ///
    /// Separated so it can be tested: this is the part S2 would have to
    /// reimplement inside `li-stream`, and it is the part whose thresholds the
    /// flap column exists to choose.
    fn fold(&mut self, ends: &[usize], n: usize, now: Duration) {
        for (&i, t) in self.seen.iter_mut() {
            if ends.contains(&i) {
                t.streak += 1;
                t.present = true;
                // Two runs agreeing, and a word already decoded after it: the
                // model has seen what follows and still says the sentence
                // ended there. `li_stream::agree` commits on the same evidence,
                // and the second half of the rule is also what stage S2 needs
                // to get a real `t_end` -- a mark with nothing after it is the
                // model guessing the utterance is over.
                if t.stable.is_none() && t.streak >= 2 && i + 1 < n {
                    t.stable = Some(now);
                }
            } else {
                if t.present {
                    t.flaps += 1;
                }
                t.present = false;
                t.streak = 0;
            }
        }
        for &i in ends {
            self.seen.entry(i).or_insert(Timing {
                first: now,
                stable: None,
                flaps: 0,
                present: true,
                streak: 1,
            });
        }
    }

    /// Close the line and record every inner boundary its final text kept.
    ///
    /// `t_close` is where the `Final` arrived, not `t_end`: the difference is
    /// the 0.6 s of trailing silence the endpoint detector waits out, and that
    /// silence is precisely what this feature would stop waiting for.
    fn close(&mut self, words: &[Word], t_close: Duration) {
        for (i, w) in words.iter().enumerate() {
            // The boundary on the last word is the endpoint doing its job; it
            // is worth nothing to predict, for the same reason `inner` exists.
            if i + 1 >= words.len() || !ends_sentence(&w.text) {
                continue;
            }
            self.rows.push(match self.seen.get(&i) {
                Some(t) => Early {
                    saved: t.stable.map(|s| t_close.saturating_sub(s).as_secs_f64()),
                    saved_raw: Some(t_close.saturating_sub(t.first).as_secs_f64()),
                    lag: t.stable.map(|s| s.as_secs_f64() - w.end.as_secs_f64()),
                    flaps: t.flaps,
                },
                // Present when the line closed but in none of its partials:
                // the endpoint re-decode changed the words. Counted, because a
                // boundary that only exists in hindsight cannot be cut on.
                None => Early {
                    saved: None,
                    saved_raw: None,
                    lag: None,
                    flaps: 0,
                },
            });
        }
        self.seen.clear();
    }
}

/// Does this already-split word end a sentence?
///
/// The mark may sit behind a closing quote or bracket, which is how
/// `you."` hides a full stop.
fn ends_sentence(w: &str) -> bool {
    w.trim_end_matches(['"', '\'', ')', ']', '\u{201d}'])
        .ends_with(MARKS)
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
        let mark = ends_sentence(raw);
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

fn report(args: &Args, utts: &[Utterance], mismatches: u64, watch: &Watch) {
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

    e2(watch);

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

/// The E2 table: how much of the wait the boundary was knowable for.
fn e2(w: &Watch) {
    if w.calls == 0 {
        return;
    }
    let n = w.rows.len();
    let unseen = w.rows.iter().filter(|r| r.saved_raw.is_none()).count();
    let unstable = w
        .rows
        .iter()
        .filter(|r| r.saved_raw.is_some() && r.saved.is_none())
        .count();
    let col = |f: fn(&Early) -> Option<f64>| {
        let mut v: Vec<f64> = w.rows.iter().filter_map(f).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let saved = col(|r| r.saved);
    let raw = col(|r| r.saved_raw);
    let lag = col(|r| r.lag);
    let flaps = col(|r| Some(r.flaps as f64));

    println!("\n## E2 — how early was the boundary knowable\n");
    println!("| | n | p50 | p90 | max |");
    println!("|---|---|---|---|---|");
    println!(
        "| **`t_close − t_stable`** (s) | {} | **{:.2}** | {:.2} | {:.2} |",
        saved.len(),
        pct(&saved, 0.5),
        pct(&saved, 0.9),
        pct(&saved, 1.0)
    );
    println!(
        "| `t_close − t_first` (s) | {} | {:.2} | {:.2} | {:.2} |",
        raw.len(),
        pct(&raw, 0.5),
        pct(&raw, 0.9),
        pct(&raw, 1.0)
    );
    println!(
        "| `t_stable − t_boundary` (s) | {} | {:.2} | {:.2} | {:.2} |",
        lag.len(),
        pct(&lag, 0.5),
        pct(&lag, 0.9),
        pct(&lag, 1.0)
    );
    println!(
        "| flaps before close | {} | {:.0} | {:.0} | {:.0} |",
        flaps.len(),
        pct(&flaps, 0.5),
        pct(&flaps, 0.9),
        pct(&flaps, 1.0)
    );
    println!(
        "| partial restore (ms) | {} | {:.1} | {:.1} | {:.1} |",
        w.ms.len(),
        pct(&sorted(&w.ms), 0.5),
        pct(&sorted(&w.ms), 0.9),
        pct(&sorted(&w.ms), 1.0)
    );
    println!(
        "\ninner boundaries at close **{n}** · never seen in a partial **{unseen}** \
         · seen but never stable **{unstable}** · partials punctuated {} \
         (word count changed on {}, skipped)",
        w.calls, w.skipped
    );

    // The plan's gate, printed rather than left to be looked up: below a
    // second, the endpoint detector was going to fire about then anyway and
    // neither the early translation nor stage S2 pays for itself.
    let med = pct(&saved, 0.5);
    if saved.is_empty() || med.is_nan() {
        println!(
            "\n**No boundary ever became stable.** Nothing to gain here; the marks \
             this clip produced inside a line never survived two consecutive partials."
        );
    } else if med >= 1.0 {
        println!(
            "\n**Gate passed**: median {med:.2} s ≥ 1.0 s. The wait is real and the \
             boundary is knowable for most of it."
        );
    } else {
        println!(
            "\n**Gate failed**: median {med:.2} s < 1.0 s. The endpoint detector was \
             going to fire about then anyway — layer 1 only."
        );
    }
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v
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

    /// One partial is never enough: the model has not yet seen what follows.
    #[test]
    fn a_boundary_needs_two_runs_and_a_word_after_it() {
        let mut w = Watch::default();
        w.fold(&[3], 6, Duration::from_secs(1));
        assert!(w.seen[&3].stable.is_none(), "one sighting is a guess");
        w.fold(&[3], 6, Duration::from_secs(2));
        assert_eq!(w.seen[&3].stable, Some(Duration::from_secs(2)));
        assert_eq!(w.seen[&3].first, Duration::from_secs(1));
    }

    #[test]
    fn a_mark_on_the_last_decoded_word_is_the_model_guessing() {
        // Twice in a row, but nothing decoded after it -- which is exactly the
        // case where S2 would also have no word onset to cut at.
        let mut w = Watch::default();
        w.fold(&[4], 5, Duration::from_secs(1));
        w.fold(&[4], 5, Duration::from_secs(2));
        assert!(w.seen[&4].stable.is_none());
        // One more word arrives and still says the sentence ended there.
        w.fold(&[4], 6, Duration::from_secs(3));
        assert_eq!(w.seen[&4].stable, Some(Duration::from_secs(3)));
    }

    #[test]
    fn a_mark_that_comes_and_goes_is_counted_once_per_disappearance() {
        let mut w = Watch::default();
        w.fold(&[2], 9, Duration::from_secs(1));
        w.fold(&[], 9, Duration::from_secs(2));
        w.fold(&[], 9, Duration::from_secs(3));
        assert_eq!(w.seen[&2].flaps, 1, "gone is gone, not gone twice");
        // ...and the streak restarts, so it has to earn stability again.
        w.fold(&[2], 9, Duration::from_secs(4));
        assert!(w.seen[&2].stable.is_none());
        w.fold(&[2], 9, Duration::from_secs(5));
        assert_eq!(w.seen[&2].stable, Some(Duration::from_secs(5)));
    }

    #[test]
    fn closing_ignores_the_boundary_on_the_last_word() {
        let word = |text: &str, end: u64| Word {
            text: text.to_owned(),
            start: Duration::from_secs(end - 1),
            end: Duration::from_secs(end),
        };
        let mut w = Watch::default();
        w.fold(&[1], 4, Duration::from_secs(2));
        w.fold(&[1], 4, Duration::from_secs(3));
        w.close(
            &[
                word("so", 1),
                word("far.", 2),
                word("and", 3),
                word("then.", 4),
            ],
            Duration::from_secs(9),
        );
        assert_eq!(
            w.rows.len(),
            1,
            "the final `then.` is the endpoint's own cut"
        );
        assert_eq!(w.rows[0].saved, Some(6.0));
        assert_eq!(w.rows[0].lag, Some(1.0), "stable at 3 s, word ended at 2 s");
        assert!(w.seen.is_empty(), "the next line starts clean");
    }

    #[test]
    fn a_prediction_nowhere_near_a_truth_costs_precision() {
        let mut s = Score::default();
        s.score(&[(2, false)], &[9], |_| true);
        assert_eq!((s.hits, s.predicted, s.truth), (0, 1, 1));
        assert!(s.precision().abs() < f64::EPSILON);
    }
}
