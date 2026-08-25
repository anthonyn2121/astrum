//! `forge-web-export` — bake a telemetry bag into a single JSON document the
//! browser renderer can open offline (no sim, no bridge, no network).
//!
//! Config (args / env):
//! * arg 1 or `$FORGE_BAG` — input bag path (default `forge.bag`).
//! * arg 2 or `$FORGE_WEB_EXPORT` — output path (default `<bag>.json`; `-` =
//!   stdout).
//!
//! The output shape is [`forge_web::ExportDoc`]; open it in the renderer with
//! its "Open file" control.

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use forge_proto::BagReader;
use forge_web::ExportDoc;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn main() -> Result<(), Box<dyn Error>> {
    let input = std::env::args().nth(1).unwrap_or_else(|| env_or("FORGE_BAG", "forge.bag"));
    let output = std::env::args()
        .nth(2)
        .unwrap_or_else(|| env_or("FORGE_WEB_EXPORT", &format!("{input}.json")));

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
