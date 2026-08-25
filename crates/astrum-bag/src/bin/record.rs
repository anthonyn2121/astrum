//! `astrum-bag-record` — subscribe to the live telemetry stream and write every
//! sample to a bag file.
//!
//! Config (CLI arg > env > default): see `--help`. Each flag falls back to its
//! `ASTRUM_*` env var, so compose/CI keep working.

use std::error::Error;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use astrum_bag::BagWriter;
use astrum_net::{NetError, Session, TelemetrySubscriber};
use clap::Parser;

/// Record the live telemetry stream to a bag file.
#[derive(Parser)]
#[command(name = "astrum-bag-record", version, about)]
struct Cli {
    /// Output bag path.
    #[arg(env = "ASTRUM_BAG", default_value = "astrum.bag")]
    out: PathBuf,

    /// Zenoh endpoint to connect to (the sim or a replay).
    #[arg(long, env = "ASTRUM_ZENOH_CONNECT", default_value_t = astrum_net::DEFAULT_CONNECT_ENDPOINT.to_string())]
    connect: String,

    /// Wall-clock seconds to record, then stop; 0 = until killed (frames are
    /// flushed as they arrive, so a kill loses nothing).
    #[arg(long, env = "ASTRUM_BAG_SECONDS", default_value_t = 0.0)]
    seconds: f64,
}

fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let out = cli.out.display().to_string();
    let endpoint = cli.connect;
    let seconds = cli.seconds;

    let session = Session::connect(&endpoint)?;
    let mut sub = TelemetrySubscriber::connect(&session)?;
    sub.subscribe_all()?;
    // Timeout so we can honour the duration limit even when no data arrives.
    sub.set_recv_timeout_ms(500);

    let mut bag = BagWriter::create(&out)?;
    if seconds > 0.0 {
        println!("[record] {endpoint} → {out} (for {seconds:.1}s)");
    } else {
        println!("[record] {endpoint} → {out} (until killed)");
    }

    let start = Instant::now();
    let mut n: u64 = 0;
    loop {
        if seconds > 0.0 && start.elapsed().as_secs_f64() >= seconds {
            break;
        }
        match sub.recv() {
            Ok(sample) => {
                bag.write_frame(now_ns(), &sample)?;
                bag.flush()?; // durable per frame: a kill loses nothing
                n += 1;
                if n.is_multiple_of(240) {
                    println!("[record] {n} frames");
                }
            }
            Err(NetError::Timeout) => continue, // recv timeout → re-check duration
            Err(e) => {
                eprintln!("[record] recv error: {e}");
                break;
            }
        }
    }

    bag.flush()?;
    println!("[record] wrote {} frames to {out}", bag.frames());
    Ok(())
}
