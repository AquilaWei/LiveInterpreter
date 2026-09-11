//! `cargo xtask mt` — translate a file of finalised source lines and score it.
//!
//! ```text
//! cargo xtask mt --lines <src.txt> [--ref <zh.txt>] [--beam N] [--chunk N] [--run N] [--no-trim]
//! ```
//!
//! The input is one finalised source line per line: exactly what `li-stream`
//! emits as `SourceFinal` and what `li-core` will hand to the translator. This
//! is how the settings in `li_mt::NllbConfig` were chosen, and it is the half
//! of gate G5 that can be measured without a person.
//!
//! Two numbers come out:
//!
//! * **chrF** against a reference translation. Every `testdata/*.zh.txt` was
//!   machine-written and is not human-verified (`testdata/README.md`), so this
//!   compares *settings against each other* and its absolute value is not a
//!   quality claim.
//! * **lines cut short** — the share of translations that end on a comma, i.e.
//!   mid-clause. This needs no reference at all, and it is the number that
//!   exposed NLLB dropping clauses in the first place.

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use li_mt::{LocalNllb, NllbConfig, Zh};

use crate::metrics::chrf;
use li_mt::chunk::Marks;

pub struct Args {
    pub lines: PathBuf,
    pub reference: Option<PathBuf>,
    pub cfg: NllbConfig,
    /// Score the model's own output, before OpenCC and punctuation.
    pub raw: bool,
    /// Whether a recogniser heard this file's punctuation or a model restored
    /// it. Defaults to heard, which is what every file scored before task 1.25
    /// was -- an accurate-lane transcript or a reference.
    pub marks: Marks,
    pub out: Option<PathBuf>,
}

pub fn parse(mut args: impl Iterator<Item = String>) -> Result<Args> {
    let (mut lines, mut reference, mut out, mut raw) = (None, None, None, false);
    let mut marks = Marks::Heard;
    let mut cfg = NllbConfig::default();
    while let Some(flag) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--lines" => lines = Some(PathBuf::from(val()?)),
            "--ref" => reference = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--model" => cfg.model_dir = PathBuf::from(val()?),
            "--beam" => cfg.beam_size = val()?.parse()?,
            "--chunk" => cfg.max_chunk_words = val()?.parse()?,
            "--run" => cfg.max_run_words = val()?.parse()?,
            "--threads" => cfg.threads = val()?.parse()?,
            "--no-trim" => cfg.trim_final_stop = false,
            "--raw" => raw = true,
            "--marks" => {
                marks = match val()?.as_str() {
                    "heard" => Marks::Heard,
                    "restored" => Marks::Restored,
                    other => bail!("--marks wants heard|restored, not {other}"),
                }
            }
            other => bail!("unknown flag {other}"),
        }
    }
    Ok(Args {
        lines: lines.context("--lines <file> is required")?,
        reference,
        cfg,
        raw,
        marks,
        out,
    })
}

pub fn run(args: Args) -> Result<()> {
    let text = std::fs::read_to_string(&args.lines)
        .with_context(|| format!("reading {}", args.lines.display()))?;
    let srcs: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if srcs.is_empty() {
        bail!("{} has no lines", args.lines.display());
    }

    let t0 = Instant::now();
    let mt = LocalNllb::open(&args.cfg)?;
    let zh = Zh::new()?;
    let load = t0.elapsed();

    let (mut outs, mut took, mut cut_short) = (Vec::new(), Vec::new(), 0usize);
    for src in &srcs {
        let t = Instant::now();
        let pieces = mt.pieces(src, args.marks)?;
        let line = if args.raw {
            zh.to_tw(&pieces.join(" "))
        } else {
            zh.finish(&pieces)
        };
        took.push(t.elapsed());
        // Measured before `Zh::finish` closes it, or the metric measures the
        // post-processing rather than the model.
        if pieces
            .last()
            .is_some_and(|p| p.trim_end().ends_with([',', '，', '、']))
        {
            cut_short += 1;
        }
        outs.push(line);
    }
    took.sort_unstable();

    let mut report = String::new();
    let mut w = |s: String| report.push_str(&s);
    w(format!("# xtask mt — {}\n\n", args.lines.display()));
    w(format!(
        "beam {} · clause split >{} words · width cut >{} words ({} marks) · trim final stop {} · {} threads{}\n\n",
        args.cfg.beam_size,
        args.cfg.max_chunk_words,
        args.cfg.max_run_words,
        match args.marks {
            Marks::Heard => "heard",
            Marks::Restored => "restored",
        },
        if args.cfg.trim_final_stop {
            "on"
        } else {
            "off"
        },
        args.cfg.threads,
        if args.raw {
            " · scored before OpenCC/punctuation"
        } else {
            ""
        },
    ));

    if let Some(r) = &args.reference {
        let reference =
            std::fs::read_to_string(r).with_context(|| format!("reading {}", r.display()))?;
        w(format!(
            "| chrF vs `{}` | **{:.1}** |\n",
            r.display(),
            chrf(reference.trim(), &outs.concat())
        ));
        w("|---|---|\n".to_string());
    } else {
        w("| | |\n|---|---|\n".to_string());
    }
    let pick = |q: f64| {
        took.get(((took.len() as f64 * q) as usize).min(took.len() - 1))
            .copied()
            .unwrap_or_default()
    };
    w(format!(
        "| lines cut short | {:.1}% ({cut_short}/{}) |\n",
        100.0 * cut_short as f64 / srcs.len() as f64,
        srcs.len()
    ));
    w(format!("| median | {} |\n", ms(pick(0.5))));
    w(format!("| p90 | {} |\n", ms(pick(0.9))));
    w(format!("| max | {} |\n", ms(pick(1.0))));
    w(format!("| model load | {:.2} s |\n", load.as_secs_f64()));
    w(format!(
        "| peak RSS | {} MiB |\n",
        peak_rss_mib().unwrap_or(0)
    ));

    w("\n## Translations\n\n".to_string());
    for (src, zh) in srcs.iter().zip(&outs) {
        w(format!("- {src}\n  {zh}\n"));
    }

    print!("{report}");
    if let Some(p) = &args.out {
        std::fs::write(p, &report).with_context(|| format!("writing {}", p.display()))?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}

fn ms(d: Duration) -> String {
    format!("{:.0} ms", d.as_secs_f64() * 1e3)
}

/// `VmHWM`: the high-water mark, which is what gate G4 is about.
fn peak_rss_mib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|l| {
            l.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
                .map(|kb| kb / 1024)
        })
}
