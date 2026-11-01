//! `astrum-web` — stream live/replayed telemetry to the browser and serve the
//! static 3-D renderer, on one port.
//!
//!  * **`GET /telemetry`** (WebSocket) — one JSON message per fleet snapshot,
//!    `{"type":"frames","frames":[FrameJson…]}`, at most `--max-hz` per second.
//!    The upstream is the Zenoh telemetry stream, so the source is either the
//!    live sim or `astrum-bag-play`. Upgrades from another origin are refused.
//!  * **`GET /tiles/<z>/<x>/<y>.jpg`** — slippy-map tile proxy with a disk cache.
//!  * **`GET /…`** — static files from `--root`.
//!
//! Config (CLI arg > env > default): see `--help`.

use std::error::Error;
use std::future::IntoFuture;
use std::path::PathBuf;
use std::time::Duration;

use astrum_net::{LatestTelemetry, Selection, Session};
use astrum_web::server::{self, Config};
use clap::Parser;
use tokio::sync::{oneshot, watch};

/// Stream live/replayed telemetry to browsers over WebSocket and serve the
/// static 3-D renderer, both on one port.
#[derive(Parser)]
#[command(name = "astrum-web", version, about)]
struct Cli {
    /// Zenoh endpoint to connect to (the sim or a replay).
    #[arg(long, env = "ASTRUM_ZENOH_CONNECT", default_value = astrum_net::DEFAULT_CONNECT_ENDPOINT)]
    connect: String,

    /// Seconds to wait for the sim (or replay) to accept the connection.
    #[arg(long, env = "ASTRUM_CONNECT_TIMEOUT_S", default_value_t = 60.0)]
    connect_timeout: f64,

    /// HTTP/WebSocket listen address (loopback by default; pass 0.0.0.0:8080 to
    /// expose it, e.g. inside a container).
    #[arg(long, env = "ASTRUM_WEB_BIND", default_value = "127.0.0.1:8080")]
    bind: String,

    /// Static renderer directory to serve.
    #[arg(long, env = "ASTRUM_WEB_ROOT", default_value = "web")]
    root: PathBuf,

    /// Most fleet snapshots per second sent to each browser.
    #[arg(long, env = "ASTRUM_WEB_MAX_HZ", default_value_t = 30.0)]
    max_hz: f64,

    /// Upstream slippy-map tile template (`{z}`/`{x}`/`{y}` placeholders); the
    /// renderer's ground map is proxied + cached through `/tiles/`. Default: Esri
    /// World Imagery (global aerial, no key). Empty = serve the cache only.
    #[arg(
        long,
        env = "ASTRUM_TILE_URL",
        default_value = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )]
    tile_url: String,

    /// On-disk cache directory for proxied map tiles (created on first miss). A
    /// tile fetched once while online is served from here afterwards (offline).
    #[arg(long, env = "ASTRUM_TILE_CACHE", default_value = "tilecache")]
    tile_cache: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    if !(cli.max_hz.is_finite() && cli.max_hz > 0.0) {
        return Err(format!("--max-hz must be a positive number, got {}", cli.max_hz).into());
    }
    let min_period = Duration::from_secs_f64(1.0 / cli.max_hz);
    let wait = Duration::try_from_secs_f64(cli.connect_timeout)
        .map_err(|e| format!("--connect-timeout {}: {e}", cli.connect_timeout))?;

    // Zenoh's blocking API runs on its own runtime: drive it from a plain thread,
    // outside this one. It reports why it stopped through `stopped`.
    let (tx, rx) = watch::channel(None);
    let (stopped_tx, stopped) = oneshot::channel::<String>();
    let connect = cli.connect.clone();
    std::thread::Builder::new().name("telemetry".into()).spawn(move || {
        let result = Session::connect(&connect, wait).and_then(|session| {
            let mut latest = LatestTelemetry::subscribe(&session, Selection::Fleet)?;
            server::pump_snapshots(&mut latest, &tx, min_period)
        });
        let why = match result {
            Ok(()) => "telemetry pump stopped".to_owned(),
            Err(e) => format!("telemetry from {connect}: {e}"),
        };
        let _ = stopped_tx.send(why);
    })?;

    let cfg = Config { root: cli.root, tile_url: cli.tile_url, tile_cache: cli.tile_cache };
    let tiles = if cfg.tile_url.is_empty() {
        format!("cache-only ({})", cfg.tile_cache.display())
    } else {
        format!("{} → cache {}", cfg.tile_url, cfg.tile_cache.display())
    };
    let root = cfg.root.display().to_string();
    let app = server::router(cfg, rx)?;

    let listener = tokio::net::TcpListener::bind(&cli.bind)
        .await
        .map_err(|e| format!("binding {}: {e}", cli.bind))?;
    let addr = listener.local_addr()?;
    println!("[astrum-web] serving {root} on http://{addr}");
    println!(
        "[astrum-web] telemetry WebSocket at ws://{addr}/telemetry (upstream {}, ≤{} Hz)",
        cli.connect, cli.max_hz
    );
    println!("[astrum-web] map tiles: {tiles}");

    let serve = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());
    tokio::select! {
        served = serve.into_future() => Ok(served?),
        why = stopped => Err(why.unwrap_or_else(|_| "telemetry thread panicked".to_owned()).into()),
    }
}

/// Resolves on Ctrl-C or SIGTERM (`docker stop`).
async fn shutdown_signal() {
    let interrupt = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                term.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = interrupt => {}
        () = terminate => {}
    }
}
