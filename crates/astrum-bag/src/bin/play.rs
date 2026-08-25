//! `astrum-bag-play` — replay a bag file as a telemetry `PUB`, preserving the
//! recorded inter-frame timing. Subscribers (controllers, the replay renderer)
//! receive it exactly as they would the live sim.
//!
//! Config (CLI arg > env > default): see `--help`. Each flag falls back to its
//! `ASTRUM_*` env var, so compose/CI keep working.

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;

use astrum_bag::BagReader;
use astrum_net::{Session, TelemetryPublisher, DEFAULT_LISTEN_ENDPOINT};
use clap::Parser;

/// Replay a bag file as a telemetry publisher.
#[derive(Parser)]
#[command(name = "astrum-bag-play", version, about)]
struct Cli {
    /// Input bag path.
    #[arg(env = "ASTRUM_BAG", default_value = "astrum.bag")]
    input: PathBuf,

    /// Zenoh endpoint to publish on (defaults to the live sim endpoint, so
    /// replay is a drop-in for the sim).
    #[arg(long, env = "ASTRUM_ZENOH_LISTEN", default_value_t = DEFAULT_LISTEN_ENDPOINT.to_string())]
    listen: String,

    /// Playback speed multiplier (2.0 = twice real-time; 0 or negative = no
    /// pacing / as fast as possible).
    #[arg(long, env = "ASTRUM_REPLAY_RATE", default_value_t = 1.0)]
    rate: f64,

    /// Pause after opening the session so subscribers can connect before
    /// playback starts (a subscription-propagation window), in milliseconds.
    #[arg(long, env = "ASTRUM_REPLAY_WARMUP_MS", default_value_t = 300)]
    warmup_ms: u64,
}

/// Cap a single inter-frame sleep so a large gap in the recording (or a paused
/// sim) doesn't stall replay indefinitely.
const MAX_GAP: Duration = Duration::from_secs(5);

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let input = cli.input.display().to_string();
    let bind = cli.listen;
    let rate = cli.rate;
    let warmup_ms = cli.warmup_ms;

    let session = Session::listen(&bind)?;
    let publisher = TelemetryPublisher::new(&session);
    println!("[play] {input} → publish on {bind} (rate {rate}x)");

    // Let subscribers connect before the first frame goes out.
    sleep(Duration::from_millis(warmup_ms));

    let file = File::open(&input).map_err(|e| format!("opening bag {input}: {e}"))?;
    let mut reader = BagReader::new(BufReader::new(file));

    let mut prev_ns: Option<u64> = None;
    let mut n: u64 = 0;
    while let Some((recv_wall_ns, sample)) = reader.next()? {
        // Reproduce the recorded gap between frames, scaled by `rate`.
        if rate > 0.0 {
            if let Some(prev) = prev_ns {
                let gap_ns = recv_wall_ns.saturating_sub(prev) as f64 / rate;
                if gap_ns > 0.0 {
                    sleep(Duration::from_nanos(gap_ns as u64).min(MAX_GAP));
                }
            }
        }
        prev_ns = Some(recv_wall_ns);
        publisher.publish(&sample)?;
        n += 1;
    }

    println!("[play] replayed {n} frames");
    Ok(())
}
