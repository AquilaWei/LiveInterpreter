//! `cargo xtask <task>` — build and measurement chores that do not belong in
//! the library crates.

use anyhow::{Result, bail};

mod eval;
mod metrics;
mod mt;
mod pause;
mod punct;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("eval") => eval::run(eval::parse(args)?),
        Some("mt") => mt::run(mt::parse(args)?),
        Some("pause") => pause::run(pause::parse(args)?),
        Some("punct") => punct::run(punct::parse(args)?),
        Some("score") => score(args),
        Some("package-models") => bail!("`xtask package-models` lands with task 1.13"),
        Some(other) => bail!("unknown task `{other}`"),
        None => {
            eprintln!("usage: cargo xtask <eval|mt|pause|punct|score|package-models>");
            Ok(())
        }
    }
}

/// Score a transcript that already exists, without re-running the pipeline.
///
/// Kept separate from `eval` so a report from an earlier task can be re-scored
/// with today's metric, and so the Rust port can be checked against
/// `poc/liveinterpreter_poc/metrics.py` on the same pair of files.
fn score(mut args: impl Iterator<Item = String>) -> Result<()> {
    let (mut reference, mut hypothesis) = (None, None);
    while let Some(flag) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("a flag is missing its value"))
        };
        match flag.as_str() {
            "--ref" => reference = Some(val()?),
            "--hyp" => hypothesis = Some(val()?),
            other => bail!("unknown flag {other}"),
        }
    }
    let (Some(r), Some(h)) = (reference, hypothesis) else {
        bail!("usage: cargo xtask score --ref <file> --hyp <file>");
    };
    let m = metrics::measure(
        &std::fs::read_to_string(r)?,
        &std::fs::read_to_string(h)?,
        5,
    );
    println!(
        "wer {:.2}%  content_wer {:.2}%  drop_rate {:.2}%  ref_words {}",
        m.wer, m.content_wer, m.drop_rate, m.ref_words
    );
    for d in &m.drops {
        println!("  hole: {d}");
    }
    Ok(())
}
