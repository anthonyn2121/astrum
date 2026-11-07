//! `astrum-bag-metrics` — print a bag's run metrics (fleet-level numbers, each
//! aircraft's extremes and final state) as pretty JSON; see
//! [`astrum_bag::metrics`].
//!
//! Config (CLI arg > env > default): see `--help`. Each flag falls back to its
//! `ASTRUM_*` env var, so scripts and CI can configure them.

use std::error::Error;
use std::path::PathBuf;

use astrum_bag::{metrics, BagReader};
use clap::Parser;

/// Print a bag's run metrics as JSON.
#[derive(Parser)]
#[command(name = "astrum-bag-metrics", version, about = "Print a bag's run metrics as JSON.")]
struct Cli {
    /// Input bag path.
    #[arg(env = "ASTRUM_BAG")]
    input: PathBuf,

    /// Write the JSON to this file instead of stdout.
    #[arg(long, env = "ASTRUM_METRICS_OUT")]
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let input = cli.input.display().to_string();

    let bag = BagReader::open(&cli.input).map_err(|e| format!("opening bag {input}: {e}"))?;
    let metrics = metrics::compute(&bag).map_err(|e| format!("reading bag {input}: {e}"))?;
    let json = serde_json::to_string_pretty(&metrics)?;
    match cli.out {
        Some(out) => std::fs::write(&out, json + "\n")
            .map_err(|e| format!("writing {}: {e}", out.display()))?,
        None => println!("{json}"),
    }
    Ok(())
}
