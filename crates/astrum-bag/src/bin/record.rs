//! `astrum-bag-record` — subscribe to the live telemetry stream and write every
//! sample to a bag file.
//!
//! Config (env / args):
//! * arg 1 or `$ASTRUM_BAG` — output path (default `astrum.bag`).
//! * `$ASTRUM_TELEMETRY_CONNECT` — publisher to connect to (default `tcp://127.0.0.1:5555`).
//! * `$ASTRUM_BAG_SECONDS` — wall-clock seconds to record, then stop; `0`/unset
//!   records until the process is killed (frames are flushed as they arrive, so
//!   a kill loses nothing).

use std::error::Error;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use astrum_bag::BagWriter;
use astrum_net::{context, zmq, TelemetrySubscriber};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let out = std::env::args().nth(1).unwrap_or_else(|| env_or("ASTRUM_BAG", "astrum.bag"));
    let endpoint = env_or("ASTRUM_TELEMETRY_CONNECT", "tcp://127.0.0.1:5555");
    let seconds: f64 = env_or("ASTRUM_BAG_SECONDS", "0").parse().unwrap_or(0.0);

    let ctx = context();
    let sub = TelemetrySubscriber::connect(&ctx, &endpoint)?;
    sub.subscribe_all()?;
    // Timeout so we can honour the duration limit even when no data arrives.
    sub.set_recv_timeout_ms(500)?;

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
            Err(zmq::Error::EAGAIN) => continue, // recv timeout → re-check duration
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
