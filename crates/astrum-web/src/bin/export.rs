//! `astrum-web-export` — bake a telemetry bag into a single JSON document the
//! browser renderer can open offline (no sim, no bridge, no network).
//!
//! Config (CLI arg > env > default): see `--help`. `--out` (env
//! `$ASTRUM_WEB_EXPORT`) defaults to `<bag>.json`; `-` writes to stdout.
//!
//! The output shape is [`astrum_web::ExportDoc`]; open it in the renderer with
//! its "Open file" control.

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use astrum_proto::BagReader;
use astrum_web::ExportDoc;
use clap::Parser;

/// Bake a telemetry bag into a single JSON document for offline replay.
#[derive(Parser)]
#[command(name = "astrum-web-export", version, about)]
struct Cli {
    /// Input bag path.
    #[arg(env = "ASTRUM_BAG", default_value = "astrum.bag")]
    input: PathBuf,

    /// Output path (defaults to `<bag>.json`; `-` writes to stdout).
    #[arg(long, env = "ASTRUM_WEB_EXPORT")]
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let input = cli.input.display().to_string();
    // The output default depends on the input, so resolve it after parsing.
    let output =
        cli.out.map(|p| p.display().to_string()).unwrap_or_else(|| format!("{input}.json"));

    let file = File::open(&input).map_err(|e| format!("opening bag {input}: {e}"))?;
    let mut reader = BagReader::new(BufReader::new(file));
    let mut frames = Vec::new();
    while let Some(frame) = reader.next()? {
        frames.push(frame);
    }

    let doc = ExportDoc::from_frames(frames);
    eprintln!(
        "[export] {input}: {} frames, {} aircraft, {:.1}–{:.1}s sim",
        doc.meta.frames,
        doc.meta.aircraft.len(),
        doc.meta.sim_time_start,
        doc.meta.sim_time_end,
    );

    if output == "-" {
        serde_json::to_writer(std::io::stdout().lock(), &doc)?;
    } else {
        let mut w =
            BufWriter::new(File::create(&output).map_err(|e| format!("creating {output}: {e}"))?);
        serde_json::to_writer(&mut w, &doc)?;
        w.flush()?;
        eprintln!("[export] wrote {output}");
    }
    Ok(())
}
