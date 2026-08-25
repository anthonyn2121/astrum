//! `astrum-bag-play` — replay a bag file as a telemetry `PUB`, preserving the
//! recorded inter-frame timing. Subscribers (controllers, the future replay
//! renderer) receive it exactly as they would the live sim.
//!
//! Config (env / args):
//! * arg 1 or `$ASTRUM_BAG` — input path (default `astrum.bag`).
//! * `$ASTRUM_BAG_REPLAY_BIND` — endpoint to publish on (default the live
//!   telemetry endpoint, so replay is a drop-in for the sim).
//! * `$ASTRUM_REPLAY_RATE` — playback speed multiplier (default `1.0`; `2.0` =
//!   twice real-time, `0` or negative = no pacing / as-fast-as-possible).
//! * `$ASTRUM_REPLAY_WARMUP_MS` — pause after binding so subscribers can connect
//!   before playback starts (default `300`; PUB/SUB has a slow-joiner race).

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::thread::sleep;
use std::time::Duration;

use astrum_bag::BagReader;
use astrum_net::{context, TelemetryPublisher, DEFAULT_TELEMETRY_ENDPOINT};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Cap a single inter-frame sleep so a large gap in the recording (or a paused
/// sim) doesn't stall replay indefinitely.
const MAX_GAP: Duration = Duration::from_secs(5);

fn main() -> Result<(), Box<dyn Error>> {
    let input = std::env::args().nth(1).unwrap_or_else(|| env_or("ASTRUM_BAG", "astrum.bag"));
    let bind = env_or("ASTRUM_BAG_REPLAY_BIND", DEFAULT_TELEMETRY_ENDPOINT);
    let rate: f64 = env_or("ASTRUM_REPLAY_RATE", "1.0").parse().unwrap_or(1.0);
    let warmup_ms: u64 = env_or("ASTRUM_REPLAY_WARMUP_MS", "300").parse().unwrap_or(300);

    let ctx = context();
    let publisher = TelemetryPublisher::bind(&ctx, &bind)?;
    println!("[play] {input} → PUB {bind} (rate {rate}x)");

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
