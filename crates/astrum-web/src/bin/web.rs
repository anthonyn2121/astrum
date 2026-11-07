//! `astrum-web` — stream live/replayed telemetry to the browser and serve the
//! static 3-D renderer, on one port.
//!
//!  * **`GET /telemetry`** (WebSocket) — every aircraft's telemetry thinned to
//!    at most `--sample-hz` samples per SIM second (plus each aircraft's
//!    newest), batched as `{"type":"frames","frames":[FrameJson…]}` at most
//!    `--max-hz` times per wall second (several samples per aircraft each,
//!    time-ordered per aircraft),
//!    plus every `astrum/events` event as `{"type":"events","events":[EventJson…]}`
//!    (in order, never throttled; a browser that connects mid-run gets only the
//!    events from then on), plus the scenario's aircraft models as
//!    `{"type":"models","models":[{"id":1,"model":"f16"}…]}` (on connect and
//!    whenever they change), plus `{"type":"run","run":N,"endpoint":"…"}` on
//!    connect and whenever the bridge is attached elsewhere (everything after
//!    it belongs to the new run). The upstream is the Zenoh stream, so the
//!    source is either the live sim or `astrum-bag-play`. Upgrades from another
//!    origin, or naming a `Host` other than `localhost`, an IP address or one
//!    of `--allowed-hosts`, are refused.
//!  * **`/api/…`** — the JSON API (`astrum_web::api`): bridge status and run
//!    control, behind the same `Host` check.
//!  * **`GET /tiles/<z>/<x>/<y>.jpg`** — slippy-map tile proxy with a disk cache.
//!  * **`GET /…`** — static files from `--root`.
//!
//! The bridge attaches to `--connect` at startup and keeps retrying in the
//! background until a sim answers there; the server runs whether or not one
//! ever does.
//!
//! Config (CLI arg > env > default): see `--help`.

use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;

use astrum_web::bridge::{Bridge, BridgeConfig};
use astrum_web::catalogs::Catalogs;
use astrum_web::server::{self, Config};
use clap::Parser;

/// How often the scenario is re-fetched, so a restarted sim or replay with
/// another fleet reaches the browsers within a few seconds.
const MODELS_POLL: Duration = Duration::from_secs(5);

/// Stream live/replayed telemetry to browsers over WebSocket and serve the
/// static 3-D renderer, both on one port.
#[derive(Parser)]
#[command(name = "astrum-web", version, about)]
struct Cli {
    /// Zenoh endpoint of the sim (or replay) to attach to at startup; the
    /// bridge waits for one to answer there.
    #[arg(long, env = "ASTRUM_ZENOH_CONNECT", default_value = astrum_net::DEFAULT_CONNECT_ENDPOINT)]
    connect: String,

    /// HTTP/WebSocket listen address (loopback by default; pass 0.0.0.0:8080 to
    /// expose it, e.g. inside a container).
    #[arg(long, env = "ASTRUM_WEB_BIND", default_value = "127.0.0.1:8080")]
    bind: String,

    /// Static renderer directory to serve.
    #[arg(long, env = "ASTRUM_WEB_ROOT", default_value = "web")]
    root: PathBuf,

    /// Most sample batches per (wall) second sent to each browser.
    #[arg(long, env = "ASTRUM_WEB_MAX_HZ", default_value_t = 30.0)]
    max_hz: f64,

    /// Most telemetry samples per aircraft per SIM second sent to each browser
    /// (each batch also carries every aircraft's newest sample), so trails stay
    /// smooth when the sim runs faster than real time.
    #[arg(long, env = "ASTRUM_WEB_SAMPLE_HZ", default_value_t = 30.0)]
    sample_hz: f64,

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

    /// Extra host names (comma-separated, no port) the telemetry WebSocket and
    /// `/api` answer for, besides `localhost` and IP addresses — e.g. the
    /// machine's DNS name when browsing to it from elsewhere. Any other `Host`
    /// is refused, so a rebound attacker domain cannot reach them.
    #[arg(long, env = "ASTRUM_WEB_ALLOWED_HOSTS", value_delimiter = ',')]
    allowed_hosts: Vec<String>,

    /// Config directory: the controller catalog (`controllers.toml`, read at
    /// startup) and the scenarios (`scenario.<name>.toml`) the studio offers.
    #[arg(long, env = "ASTRUM_CONFIG_DIR", default_value = "config")]
    config_dir: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let period = |flag: &str, hz: f64| -> Result<Duration, String> {
        Some(hz)
            .filter(|hz| hz.is_finite() && *hz > 0.0)
            .and_then(|hz| Duration::try_from_secs_f64(1.0 / hz).ok())
            .ok_or_else(|| format!("{flag} must be a positive number, got {hz}"))
    };
    let min_period = period("--max-hz", cli.max_hz)?;
    let sample_period = period("--sample-hz", cli.sample_hz)?;
    let mut allowed_hosts = Vec::new();
    for host in cli.allowed_hosts.iter().map(|h| h.trim()).filter(|h| !h.is_empty()) {
        if !host.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-') {
            return Err(format!(
                "--allowed-hosts: {host:?} is not a bare host name (letters, digits, '.', '-'; \
                 no port)"
            )
            .into());
        }
        allowed_hosts.push(host.to_owned());
    }
    let catalogs = Catalogs::load(&cli.config_dir).map_err(|e| e.to_string())?;
    let bridge = Bridge::new(BridgeConfig { min_period, sample_period, models_poll: MODELS_POLL });
    bridge.attach(&cli.connect);

    let cfg = Config {
        root: cli.root,
        tile_url: cli.tile_url,
        tile_cache: cli.tile_cache,
        allowed_hosts,
        catalogs,
    };
    let tiles = if cfg.tile_url.is_empty() {
        format!("cache-only ({})", cfg.tile_cache.display())
    } else {
        format!("{} → cache {}", cfg.tile_url, cfg.tile_cache.display())
    };
    let root = cfg.root.display().to_string();
    let app = server::router(cfg, bridge)?;

    let listener = tokio::net::TcpListener::bind(&cli.bind)
        .await
        .map_err(|e| format!("binding {}: {e}", cli.bind))?;
    let addr = listener.local_addr()?;
    println!("[astrum-web] serving {root} on http://{addr}");
    println!(
        "[astrum-web] telemetry WebSocket at ws://{addr}/telemetry (attached to {}, ≤{} \
         samples per sim second in ≤{} batches per second)",
        cli.connect, cli.sample_hz, cli.max_hz
    );
    println!("[astrum-web] map tiles: {tiles}");

    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;
    Ok(())
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
