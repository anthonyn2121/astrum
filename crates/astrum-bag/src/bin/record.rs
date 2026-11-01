//! `astrum-bag-record` — record the live telemetry, command and event streams
//! to an MCAP bag.
//!
//! Stops after `--seconds` or on SIGINT / SIGTERM, and always finishes the bag
//! (summary + footer) on the way out. A SIGKILL leaves an unfinished file.
//!
//! Config (CLI arg > env > default): see `--help`. Each flag falls back to its
//! `ASTRUM_*` env var, so compose/CI keep working.

use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use astrum_bag::{BagWriter, Recorder};
use astrum_net::{fetch_scenario, Session, DEFAULT_CONNECT_ENDPOINT};
use clap::Parser;

/// How long to wait for the sim's scenario before recording without it.
const SCENARIO_TIMEOUT: Duration = Duration::from_secs(2);

/// Record the live telemetry and command streams to an MCAP bag.
#[derive(Parser)]
#[command(name = "astrum-bag-record", version, about)]
struct Cli {
    /// Output bag path.
    #[arg(env = "ASTRUM_BAG", default_value = "astrum.mcap")]
    out: PathBuf,

    /// Zenoh endpoint to connect to (the sim or a replay).
    #[arg(long, env = "ASTRUM_ZENOH_CONNECT", default_value_t = DEFAULT_CONNECT_ENDPOINT.to_string())]
    connect: String,

    /// Seconds to wait for the sim (or replay) to accept the connection.
    #[arg(long, env = "ASTRUM_CONNECT_TIMEOUT_S", default_value_t = 60.0)]
    connect_timeout: f64,

    /// Wall-clock seconds to record, then stop; 0 = until SIGINT/SIGTERM.
    #[arg(long, env = "ASTRUM_BAG_SECONDS", default_value_t = 0.0)]
    seconds: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let out = cli.out.display().to_string();

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = Arc::clone(&interrupted);
        ctrlc::set_handler(move || interrupted.store(true, Ordering::SeqCst))?;
    }

    let wait = Duration::try_from_secs_f64(cli.connect_timeout)
        .map_err(|e| format!("--connect-timeout {}: {e}", cli.connect_timeout))?;
    let session = Session::connect(&cli.connect, wait)
        .map_err(|e| format!("connecting to {} within {wait:?}: {e}", cli.connect))?;
    // Subscribe first so nothing published while waiting for the scenario is missed.
    let recorder = Recorder::subscribe(&session)?;
    let scenario = fetch_scenario(&session, SCENARIO_TIMEOUT)?;
    if scenario.is_none() {
        eprintln!("[record] no scenario on astrum/scenario within {SCENARIO_TIMEOUT:?}; recording without it");
    }
    let mut bag = BagWriter::create(&cli.out, scenario.as_deref())
        .map_err(|e| format!("creating bag {out}: {e}"))?;

    let limit = match cli.seconds {
        s if s > 0.0 => {
            Some(Duration::try_from_secs_f64(s).map_err(|e| format!("--seconds {s}: {e}"))?)
        }
        _ => None,
    };
    match limit {
        Some(limit) => println!("[record] {} → {out} (for {limit:?})", cli.connect),
        None => println!("[record] {} → {out} (until SIGINT/SIGTERM)", cli.connect),
    }

    let start = Instant::now();
    let result = recorder.run(&mut bag, |_| {
        interrupted.load(Ordering::SeqCst) || limit.is_some_and(|limit| start.elapsed() >= limit)
    });
    // Finish the bag even when recording failed, so what was written stays readable.
    let finished = bag.finish();
    let stats = result?;
    finished.map_err(|e| format!("finishing bag {out}: {e}"))?;

    println!(
        "[record] wrote {out}: {} telemetry, {} command, {} event messages; {} undecodable samples skipped",
        stats.telemetry, stats.commands, stats.events, stats.rejected
    );
    Ok(())
}
