//! The `astrum-web` HTTP service: the telemetry WebSocket, the slippy-map tile
//! proxy, and the static renderer, all on one [`Router`].
//!
//! Telemetry fan-out is latest-value: a blocking thread runs [`pump_snapshots`],
//! which turns the newest sample of every aircraft into ONE JSON message per
//! snapshot (at most `1 / min_period` per second, and only when something
//! changed) and stores it in a [`watch`] channel. Each browser's task forwards
//! the newest stored message; a browser that falls behind simply skips to the
//! newest one, and a browser that connects mid-run gets the current snapshot at
//! once. The payload is a [`Utf8Bytes`], so handing it to N clients is N
//! reference-count bumps, not N copies.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use astrum_net::{LatestTelemetry, NetError};
use astrum_proto::Telemetry;
use axum::body::Bytes;
use axum::extract::ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as UrlPath, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, HOST, ORIGIN};
use axum::http::uri::Authority;
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tokio::sync::watch;
use tower::Layer;
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;

use crate::snapshot_message;

/// The newest encoded fleet snapshot (`None` until the first one).
pub type Snapshot = Option<Utf8Bytes>;

/// Highest slippy-map zoom the tile proxy accepts.
pub const MAX_ZOOM: u32 = 22;
/// Largest upstream tile the proxy accepts.
const MAX_TILE_BYTES: usize = 8 * 1024 * 1024;
/// Idle WebSocket keep-alive: detects browsers that vanished without a close.
const PING_INTERVAL: Duration = Duration::from_secs(15);
/// How often an idle pump re-checks whether anyone is still listening.
const IDLE_POLL: Duration = Duration::from_secs(1);

/// Server configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Static renderer directory (`web/`).
    pub root: PathBuf,
    /// Upstream tile URL template with `{z}`/`{x}`/`{y}`; empty = cache only.
    pub tile_url: String,
    /// On-disk tile cache directory.
    pub tile_cache: PathBuf,
}

#[derive(Clone)]
struct AppState {
    telemetry: watch::Receiver<Snapshot>,
    tiles: Arc<TileProxy>,
}

struct TileProxy {
    url: String,
    cache: PathBuf,
    client: reqwest::Client,
}

/// Build the service: `GET /telemetry` (WebSocket), `GET /tiles/<z>/<x>/<y>.jpg`,
/// and every other path served from `cfg.root`.
pub fn router(cfg: Config, telemetry: watch::Receiver<Snapshot>) -> reqwest::Result<Router> {
    // Bounded timeouts so an offline upstream fails fast (the renderer then
    // shows the plain grid where the tile would be).
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(20))
        .user_agent(concat!("astrum-web/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let tiles = Arc::new(TileProxy { url: cfg.tile_url, cache: cfg.tile_cache, client });
    // `ServeDir` rejects any path that is not plain components below the root
    // (`..`, absolute, percent-encoded separators).
    let static_files =
        SetResponseHeaderLayer::if_not_present(CACHE_CONTROL, HeaderValue::from_static("no-cache"))
            .layer(ServeDir::new(cfg.root));
    Ok(Router::new()
        .route("/telemetry", get(telemetry_ws))
        .route("/tiles/{*rest}", get(tile))
        .fallback_service(static_files)
        .with_state(AppState { telemetry, tiles }))
}

/// Whether a WebSocket upgrade may proceed: requests without an `Origin`
/// (non-browser clients) are allowed; a browser request must come from a page
/// served by this host, so an arbitrary web page cannot read the telemetry.
pub fn origin_allowed(headers: &HeaderMap) -> bool {
    let Some(origin) = headers.get(ORIGIN) else { return true };
    let Some(host) = headers.get(HOST) else { return false };
    match (origin.to_str(), host.to_str()) {
        (Ok(origin), Ok(host)) => same_origin(origin, host),
        _ => false,
    }
}

/// `origin` (`scheme://host[:port]`) names the same host and port as the
/// request's `Host` header. Ports default from the origin's scheme.
fn same_origin(origin: &str, host: &str) -> bool {
    let Ok(origin) = origin.parse::<Uri>() else { return false };
    let default_port = match origin.scheme_str() {
        Some("http") => 80,
        Some("https") => 443,
        _ => return false,
    };
    if origin.path_and_query().is_some_and(|p| p.as_str() != "/") {
        return false;
    }
    let (Some(o), Ok(h)) = (origin.authority(), host.parse::<Authority>()) else { return false };
    if o.as_str().contains('@') || h.as_str().contains('@') {
        return false;
    }
    o.host().eq_ignore_ascii_case(h.host())
        && o.port_u16().unwrap_or(default_port) == h.port_u16().unwrap_or(default_port)
}

async fn telemetry_ws(
    State(state): State<AppState>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    if !origin_allowed(&headers) {
        return (StatusCode::FORBIDDEN, "cross-origin WebSocket refused").into_response();
    }
    let rx = state.telemetry.clone();
    ws.on_upgrade(move |socket| stream_snapshots(socket, rx))
}

/// Forward the newest snapshot to one browser until it leaves.
async fn stream_snapshots(mut socket: WebSocket, mut rx: watch::Receiver<Snapshot>) {
    rx.mark_changed(); // send the current snapshot straight away
    let mut ping = tokio::time::interval(PING_INTERVAL);
    ping.tick().await; // the first tick is immediate
    loop {
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    break; // the pump is gone
                }
                let snapshot = rx.borrow_and_update().clone();
                if let Some(text) = snapshot {
                    if socket.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_)) | Err(_)) | None => break,
                Some(Ok(_)) => {} // the browser has nothing to say; pings are answered by the socket
            },
            _ = ping.tick() => {
                if socket.send(Message::Ping(Bytes::new())).await.is_err() {
                    break;
                }
            }
        }
    }
}

/// Publish fleet snapshots from `latest` into `tx`: at most one per
/// `min_period`, and only when the fleet state differs from the last one sent.
/// Samples that arrive while throttled are folded into the next snapshot.
///
/// Blocking — run it on a dedicated thread. Returns `Ok` once every receiver
/// is gone, or the transport error that stopped it.
pub fn pump_snapshots(
    latest: &mut LatestTelemetry,
    tx: &watch::Sender<Snapshot>,
    min_period: Duration,
) -> Result<(), NetError> {
    let mut sent: Vec<Telemetry> = Vec::new();
    let mut sent_at: Option<Instant> = None;
    loop {
        match latest.wait(IDLE_POLL) {
            Ok(()) => {}
            Err(NetError::Timeout) if tx.is_closed() => return Ok(()),
            Err(NetError::Timeout) => continue,
            Err(e) => return Err(e),
        }
        if let Some(at) = sent_at {
            let due = at + min_period;
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
        }
        // Mark what arrived while throttled as seen; the snapshot below has it.
        let _ = latest.wait(Duration::ZERO);
        let snapshot = latest.snapshot();
        if snapshot == sent {
            continue;
        }
        tx.send_replace(Some(Utf8Bytes::from(snapshot_message(&snapshot))));
        sent = snapshot;
        sent_at = Some(Instant::now());
    }
}

/// Parse + validate a `<z>/<x>/<y>[.jpg]` tile path: decimal digits only and in
/// range for the zoom, so the coordinates cannot escape the cache directory.
pub fn parse_tile(rest: &str) -> Option<(u32, u32, u32)> {
    let mut parts = rest.split('/');
    let (z, x, y) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let y = y.strip_suffix(".jpg").unwrap_or(y);
    let number = |s: &str| -> Option<u32> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        s.parse().ok()
    };
    let (z, x, y) = (number(z)?, number(x)?, number(y)?);
    if z > MAX_ZOOM {
        return None;
    }
    let n = 1u32 << z;
    (x < n && y < n).then_some((z, x, y))
}

/// Serve one tile from the cache, fetching + caching it from the upstream on a
/// miss. A miss with no reachable upstream is a 404 (the renderer skips it).
async fn tile(State(state): State<AppState>, UrlPath(rest): UrlPath<String>) -> Response {
    let Some((z, x, y)) = parse_tile(&rest) else {
        return (StatusCode::BAD_REQUEST, "bad tile path").into_response();
    };
    let tiles = &state.tiles;
    let path = tiles.cache.join(z.to_string()).join(x.to_string()).join(format!("{y}.jpg"));
    if let Ok(bytes) = tokio::fs::read(&path).await {
        return jpeg(bytes);
    }
    match tiles.fetch(z, x, y).await {
        Some(bytes) => {
            if let Err(e) = cache_tile(&path, &bytes).await {
                eprintln!("[astrum-web] caching tile {}: {e}", path.display());
            }
            jpeg(bytes)
        }
        None => (StatusCode::NOT_FOUND, "tile unavailable").into_response(),
    }
}

fn jpeg(bytes: Vec<u8>) -> Response {
    ([(CONTENT_TYPE, "image/jpeg"), (CACHE_CONTROL, "public, max-age=604800")], bytes)
        .into_response()
}

impl TileProxy {
    /// Fetch a tile from the upstream. `None` on any failure (cache-only mode,
    /// offline, non-200, empty, oversized).
    async fn fetch(&self, z: u32, x: u32, y: u32) -> Option<Vec<u8>> {
        if self.url.is_empty() {
            return None;
        }
        let url = self
            .url
            .replace("{z}", &z.to_string())
            .replace("{x}", &x.to_string())
            .replace("{y}", &y.to_string());
        let mut resp = self.client.get(&url).send().await.ok()?;
        if resp.status() != reqwest::StatusCode::OK
            || resp.content_length().is_some_and(|n| n > MAX_TILE_BYTES as u64)
        {
            return None;
        }
        let mut buf = Vec::new();
        while let Some(chunk) = resp.chunk().await.ok()? {
            if buf.len() + chunk.len() > MAX_TILE_BYTES {
                return None;
            }
            buf.extend_from_slice(&chunk);
        }
        (!buf.is_empty()).then_some(buf)
    }
}

/// Unique suffix for tile-cache temp files.
static TILE_TMP: AtomicU64 = AtomicU64::new(0);

/// Write a tile into the cache via a unique temp file + atomic rename, so a
/// concurrent reader never sees a partial tile.
async fn cache_tile(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let n = TILE_TMP.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("part-{}-{n}", std::process::id()));
    tokio::fs::write(&tmp, bytes).await?;
    if let Err(e) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    Ok(())
}
