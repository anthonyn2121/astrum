//! Helpers shared by the integration tests: pid-derived Zenoh endpoints, a
//! stand-in sim, and a server over a temp tree.

// Each test binary uses its own subset.
#![allow(dead_code)]

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use astrum_net::{EventPublisher, Session, TelemetryPublisher};
use astrum_proto::{encode_event, Event, EventKind, Telemetry};
use astrum_web::bridge::{Bridge, BridgeConfig};
use astrum_web::catalogs::Catalogs;
use astrum_web::runs::{Runs, RunsConfig};
use astrum_web::server::{router, Config};
use axum::Router;

/// A Zenoh endpoint on this process's own port block (`n` < 10), so parallel
/// tests and concurrently running test binaries do not collide. The band
/// (10000–19999) is below the Linux ephemeral range and clear of the other
/// crates' test ports (astrum-net: 20000–39999).
pub fn endpoint(n: u32) -> String {
    assert!(n < 10);
    format!("tcp/127.0.0.1:{}", 10_000 + (std::process::id() % 1_000) * 10 + n)
}

/// Fast pumps, so tests see changes quickly; the default sample rate (30 Hz).
pub fn config() -> BridgeConfig {
    BridgeConfig {
        min_period: Duration::from_millis(10),
        sample_period: Duration::from_secs_f64(1.0 / 30.0),
        models_poll: Duration::from_millis(100),
    }
}

/// Poll `cond` every 20 ms until it holds, failing with `what` after 10 s.
pub fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A stand-in sim: a Zenoh hub on an endpoint that publishes until dropped.
pub struct Sim {
    pub hub: Session,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Sim {
    /// Aircraft `id`'s telemetry (a 50 Hz sim at real time) and a
    /// `controller-up` event for it every 20 ms.
    pub fn flying(endpoint: &str, id: u32) -> Self {
        let hub = Session::listen(endpoint).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (session, stopped) = (hub.clone(), Arc::clone(&stop));
        let thread = std::thread::spawn(move || {
            let mut telemetry = TelemetryPublisher::new(&session);
            let events = EventPublisher::new(&session).unwrap();
            let mut tick = 0;
            while !stopped.load(Ordering::Relaxed) {
                tick += 1;
                let sim_time_s = tick as f64 / 50.0;
                let t = Telemetry { aircraft_id: id, tick, sim_time_s, ..Telemetry::default() };
                telemetry.publish(&t).unwrap();
                let up =
                    Event { tick, sim_time_s, kind: EventKind::ControllerUp { aircraft_id: id } };
                events.publish_raw(encode_event(&up)).unwrap();
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        Self { hub, stop, thread: Some(thread) }
    }

    /// A stand-in sim (no events) flying aircraft `ids` at `rate_hz` ticks per
    /// sim second and `speed` times real time: as wall time reaches each tick
    /// it publishes that tick's sample of every aircraft (ids ascending), with
    /// `sim_time_s = tick / rate_hz`, until dropped.
    pub fn paced(endpoint: &str, ids: &[u32], rate_hz: f64, speed: f64) -> Self {
        let hub = Session::listen(endpoint).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let (session, stopped, ids) = (hub.clone(), Arc::clone(&stop), ids.to_vec());
        let thread = std::thread::spawn(move || {
            let mut telemetry = TelemetryPublisher::new(&session);
            let start = Instant::now();
            let mut tick = 0;
            while !stopped.load(Ordering::Relaxed) {
                let due = (start.elapsed().as_secs_f64() * speed * rate_hz) as u64;
                while tick < due {
                    tick += 1;
                    let sim_time_s = tick as f64 / rate_hz;
                    for &aircraft_id in &ids {
                        let t = Telemetry { aircraft_id, tick, sim_time_s, ..Telemetry::default() };
                        telemetry.publish(&t).unwrap();
                    }
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        Self { hub, stop, thread: Some(thread) }
    }
}

impl Drop for Sim {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub const SECRET: &str = "top-secret-outside-root";

/// A server for `bridge` over a temp tree: `<tmp>/web/index.html`,
/// `<tmp>/secret.txt` (outside the static root), and a cache-only tile cache
/// holding `14/3/5.jpg`.
pub fn app(tmp: &Path, bridge: Bridge) -> Router {
    app_allowing(tmp, bridge, &[])
}

/// [`app`], also answering for the host names `allowed_hosts`.
pub fn app_allowing(tmp: &Path, bridge: Bridge, allowed_hosts: &[&str]) -> Router {
    build(tmp, bridge, allowed_hosts, repo_catalogs())
}

/// [`app`] over `catalogs` instead of the repository's `config/`.
pub fn app_with_catalogs(tmp: &Path, bridge: Bridge, catalogs: Catalogs) -> Router {
    build(tmp, bridge, &[], catalogs)
}

/// The `--connect` endpoint of [`app_with_runs`]' runs: where the bridge
/// goes back to once a run has finished (no sim ever answers there).
pub const RUNS_CONNECT: &str = "tcp/127.0.0.1:1";

/// [`app`] over `catalogs`, its live runs flown by `sim` into `<tmp>/runs`
/// (`--connect` = [`RUNS_CONNECT`]); also the run manager, for shutting it
/// down.
pub fn app_with_runs(tmp: &Path, bridge: Bridge, catalogs: Catalogs, sim: &Path) -> (Router, Runs) {
    let cfg =
        RunsConfig { dir: tmp.join("runs"), sim: sim.to_owned(), connect: RUNS_CONNECT.into() };
    let runs = Runs::new(cfg, bridge.clone());
    (build_with(tmp, bridge, &[], catalogs, runs.clone()), runs)
}

fn build(tmp: &Path, bridge: Bridge, allowed_hosts: &[&str], catalogs: Catalogs) -> Router {
    // No test of these flies a run.
    let cfg = RunsConfig {
        dir: tmp.join("runs"),
        sim: tmp.join("no-astrum-sim"),
        connect: RUNS_CONNECT.into(),
    };
    build_with(tmp, bridge.clone(), allowed_hosts, catalogs, Runs::new(cfg, bridge))
}

fn build_with(
    tmp: &Path,
    bridge: Bridge,
    allowed_hosts: &[&str],
    catalogs: Catalogs,
    runs: Runs,
) -> Router {
    let web = tmp.join("web");
    std::fs::create_dir_all(&web).unwrap();
    std::fs::write(web.join("index.html"), "<h1>astrum</h1>").unwrap();
    std::fs::write(tmp.join("secret.txt"), SECRET).unwrap();
    let cache = tmp.join("tiles");
    std::fs::create_dir_all(cache.join("14/3")).unwrap();
    std::fs::write(cache.join("14/3/5.jpg"), b"\xff\xd8jpeg").unwrap();
    let cfg = Config {
        root: web,
        tile_url: String::new(),
        tile_cache: cache,
        allowed_hosts: allowed_hosts.iter().map(|h| (*h).to_owned()).collect(),
        catalogs,
        runs,
    };
    router(cfg, bridge).unwrap()
}

/// The repository's `config/`: its controller catalog and bundled scenarios.
pub fn repo_config() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")
}

/// [`Catalogs`] over [`repo_config`].
pub fn repo_catalogs() -> Catalogs {
    Catalogs::load(repo_config()).unwrap()
}
