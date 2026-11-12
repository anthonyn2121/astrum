//! `astrum-bag-play` — replay a bag's telemetry and events as a stand-in for
//! the sim.
//!
//! Listens where the sim would, serves the recorded scenario on
//! `astrum/scenario`, and republishes every telemetry and event message with
//! the exact recorded payload, in recorded order, paced by sim time.
//! Subscribers (controllers, the web viewer) receive it as they would the live
//! sim. Recorded commands are for analysis and are not replayed.
//!
//! Config (CLI arg > env > default): see `--help`. Each flag falls back to its
//! `ASTRUM_*` env var, so scripts and CI can configure them.

use std::error::Error;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;

use astrum_bag::{replay, BagReader};
use astrum_net::{ScenarioServer, Session, DEFAULT_LISTEN_ENDPOINT};
use clap::Parser;

/// Replay a bag's telemetry as a stand-in for the sim.
#[derive(Parser)]
#[command(name = "astrum-bag-play", version, about)]
struct Cli {
    /// Input bag path.
    #[arg(env = "ASTRUM_BAG", default_value = "astrum.mcap")]
    input: PathBuf,

    /// Zenoh endpoint to listen on (defaults to the live sim endpoint, so
    /// replay is a drop-in for the sim).
    #[arg(long, env = "ASTRUM_ZENOH_LISTEN", default_value_t = DEFAULT_LISTEN_ENDPOINT.to_string())]
    listen: String,

    /// Playback speed multiplier on sim time (2.0 = twice real time; 0 or
    /// negative = unpaced, as fast as possible). A single gap is capped at 5 s.
    #[arg(long, env = "ASTRUM_REPLAY_RATE", default_value_t = 1.0)]
    rate: f64,

    /// Pause after opening the session so subscribers can connect before
    /// playback starts (a subscription-propagation window), in milliseconds.
    #[arg(long, env = "ASTRUM_REPLAY_WARMUP_MS", default_value_t = 300)]
    warmup_ms: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let input = cli.input.display().to_string();

    let bag = BagReader::open(&cli.input).map_err(|e| format!("opening bag {input}: {e}"))?;
    let session = Session::listen(&cli.listen)?;
    let _scenario = match bag.scenario() {
        Some(toml) => Some(ScenarioServer::serve(&session, toml.to_owned())?),
        None => {
            eprintln!("[play] {input} has no recorded scenario; astrum/scenario is not served");
            None
        }
    };
    println!("[play] {input} → publish on {} (rate {}x)", cli.listen, cli.rate);

    // Let subscribers connect before the first message goes out.
    sleep(Duration::from_millis(cli.warmup_ms));

    let stats = replay(&session, &bag, cli.rate)?;
    println!(
        "[play] replayed {} telemetry and {} event messages; skipped {} command and {} other messages",
        stats.telemetry, stats.events, stats.commands, stats.skipped
    );
    Ok(())
}
