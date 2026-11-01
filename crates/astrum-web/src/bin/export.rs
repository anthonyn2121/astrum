//! `astrum-web-export` — bake an MCAP bag's telemetry and events into a single
//! JSON document the browser renderer can open offline (no sim, no bridge, no
//! network).
//!
//! Config (CLI arg > env): see `--help`. The bag is positional (env
//! `$ASTRUM_BAG`); `--out` (env `$ASTRUM_WEB_EXPORT`) defaults to `<bag>.json`,
//! and `-` writes to stdout.
//!
//! The output shape is [`astrum_web::ExportDoc`]; open it in the renderer with
//! its "Open file" control.

use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use astrum_web::export::export_bag;
use clap::Parser;

/// Bake an MCAP bag's telemetry and events into a single JSON document for
/// offline replay.
#[derive(Parser)]
#[command(name = "astrum-web-export", version, about)]
struct Cli {
    /// Input bag (`.mcap`).
    #[arg(env = "ASTRUM_BAG")]
    bag: PathBuf,

    /// Output path (defaults to `<bag>.json`; `-` writes to stdout).
    #[arg(long, env = "ASTRUM_WEB_EXPORT")]
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let out = cli.out.unwrap_or_else(|| {
        let mut name = OsString::from(cli.bag.as_os_str());
        name.push(".json");
        PathBuf::from(name)
    });

    let doc =
        export_bag(&cli.bag).map_err(|e| format!("reading bag {}: {e}", cli.bag.display()))?;
    let m = &doc.meta;
    eprintln!(
        "[export] {}: {} frames, {} events, {} aircraft, {:.1}–{:.1} s sim, scenario {}, {} rejected",
        cli.bag.display(),
        m.frames,
        m.events,
        m.aircraft.len(),
        m.sim_time_start,
        m.sim_time_end,
        if m.scenario.is_some() { "embedded" } else { "absent" },
        m.rejected,
    );

    if out.as_os_str() == "-" {
        let mut w = BufWriter::new(std::io::stdout().lock());
        serde_json::to_writer(&mut w, &doc)?;
        w.flush()?;
    } else {
        let file = File::create(&out).map_err(|e| format!("creating {}: {e}", out.display()))?;
        let mut w = BufWriter::new(file);
        serde_json::to_writer(&mut w, &doc)?;
        w.flush()?;
        eprintln!("[export] wrote {}", out.display());
    }
    Ok(())
}
