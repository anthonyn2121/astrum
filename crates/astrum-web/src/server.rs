//! The `astrum-web` HTTP service: the telemetry WebSocket, the JSON [`api`],
//! the slippy-map tile proxy, and the static renderer, all on one [`Router`].
//!
//! Each browser's WebSocket task forwards the [`Bridge`]'s feeds on one
//! socket: the current run note first (on connect and on every attach or
//! detach), the newest fleet snapshot and aircraft-models message (latest
//! value: a browser that falls behind skips to the newest one, one that
//! connects mid-run gets the current ones at once), and every event batch of
//! its current run, in order. The run note always goes out before anything of
//! the run it announces.
//!
//! The WebSocket and every `/api` route answer only requests whose `Host` is
//! `localhost`, an IP literal, or one of [`Config::allowed_hosts`]
//! ([`host_allowed`]): a page on a domain an attacker re-points at this
//! server (DNS rebinding) is same-origin with it, so the `Origin` check alone
//! cannot stop it. Static files and map tiles hold nothing private and stay
//! open.
//!
//! [`api`]: crate::api

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as UrlPath, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, HOST, ORIGIN};
use axum::http::uri::Authority;
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{broadcast, watch};
use tower::Layer;
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;

use crate::bridge::{Bridge, EventBatch, RunNote, Snapshot};
use crate::catalogs::Catalogs;
use crate::runs::Runs;

/// Highest slippy-map zoom the tile proxy accepts.
pub const MAX_ZOOM: u32 = 22;
/// Largest upstream tile the proxy accepts.
const MAX_TILE_BYTES: usize = 8 * 1024 * 1024;
/// Idle WebSocket keep-alive: detects browsers that vanished without a close.
const PING_INTERVAL: Duration = Duration::from_secs(15);

/// Server configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Static renderer directory (`web/`).
    pub root: PathBuf,
    /// Upstream tile URL template with `{z}`/`{x}`/`{y}`; empty = cache only.
    pub tile_url: String,
    /// On-disk tile cache directory.
    pub tile_cache: PathBuf,
    /// Host names (no port) the WebSocket and the API answer besides
    /// `localhost` and IP literals; compared case-insensitively.
    pub allowed_hosts: Vec<String>,
    /// The config directory's controller catalog and scenarios (`/api`).
    pub catalogs: Catalogs,
    /// The live interactive runs (`/api/runs`); its bridge is the server's.
    pub runs: Runs,
}

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) bridge: Bridge,
    pub(crate) allowed_hosts: Arc<[String]>,
    pub(crate) catalogs: Arc<Catalogs>,
    pub(crate) runs: Runs,
    tiles: Arc<TileProxy>,
}

struct TileProxy {
    url: String,
    cache: PathBuf,
    client: reqwest::Client,
}

/// Build the service: `GET /telemetry` (WebSocket: `bridge`'s run notes, fleet
/// snapshots, aircraft-models messages and event batches), the JSON API under
/// `/api` ([`crate::api`]), `GET /tiles/<z>/<x>/<y>.jpg`, and every other path
/// served from `cfg.root`.
pub fn router(cfg: Config, bridge: Bridge) -> reqwest::Result<Router> {
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
    let state = AppState {
        bridge,
        allowed_hosts: cfg.allowed_hosts.into(),
        catalogs: Arc::new(cfg.catalogs),
        runs: cfg.runs,
        tiles,
    };
    // The guard wraps the whole API, fallbacks included, so no route added
    // there can escape it. Nested as a service, the API also gets `/api` and
    // `/api/` themselves (a nested router would leave `/api/` to the static
    // files).
    let api: Router = crate::api::router()
        .layer(axum::middleware::from_fn_with_state(state.clone(), crate::api::guard))
        .with_state(state.clone());
    Ok(Router::new()
        .route("/telemetry", get(telemetry_ws))
        .route("/tiles/{*rest}", get(tile))
        .nest_service("/api", api)
        .fallback_service(static_files)
        .with_state(state))
}

/// Whether the request's `Host` (port ignored) is one this server answers
/// for: `localhost`, an IP literal, or one of `allowed` (case-insensitive). A
/// request without a readable `Host` is refused.
pub fn host_allowed(headers: &HeaderMap, allowed: &[String]) -> bool {
    let Some(Ok(host)) = headers.get(HOST).map(|h| h.to_str()) else { return false };
    let Ok(host) = host.parse::<Authority>() else { return false };
    if host.as_str().contains('@') {
        return false;
    }
    let name = host.host();
    let bare = name.strip_prefix('[').and_then(|n| n.strip_suffix(']')).unwrap_or(name);
    bare.eq_ignore_ascii_case("localhost")
        || bare.parse::<IpAddr>().is_ok()
        || allowed.iter().any(|a| a.eq_ignore_ascii_case(bare))
}

/// Whether a WebSocket upgrade or a state-changing API request may proceed:
/// requests without an `Origin` (non-browser clients) are allowed; a browser
/// request must come from a page served by this host, so another site's page
/// can neither read the telemetry nor drive the sim.
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
    if !host_allowed(&headers, &state.allowed_hosts) {
        return (StatusCode::FORBIDDEN, "WebSocket refused: host not allowed").into_response();
    }
    if !origin_allowed(&headers) {
        return (StatusCode::FORBIDDEN, "cross-origin WebSocket refused").into_response();
    }
    let feeds = Feeds {
        runs: state.bridge.runs(),
        snapshots: state.bridge.snapshots(),
        models: state.bridge.models(),
        events: state.bridge.events(), // events from now on
    };
    ws.on_upgrade(move |socket| stream(socket, feeds))
}

/// One browser's subscriptions to the bridge.
struct Feeds {
    runs: watch::Receiver<RunNote>,
    snapshots: watch::Receiver<Snapshot>,
    models: watch::Receiver<Option<Utf8Bytes>>,
    events: broadcast::Receiver<EventBatch>,
}

/// Forward the run notes, the newest snapshot and models message, and the
/// current run's event batches to one browser until it leaves.
async fn stream(mut socket: WebSocket, feeds: Feeds) {
    let Feeds { mut runs, mut snapshots, mut models, mut events } = feeds;
    // Send the current run, snapshot and models straight away.
    runs.mark_changed();
    snapshots.mark_changed();
    models.mark_changed();
    let mut run = runs.borrow().run;
    let mut ping = tokio::time::interval(PING_INTERVAL);
    ping.tick().await; // the first tick is immediate
    loop {
        // Biased: a run note is published before anything of its run, so
        // checking it first sends it first.
        tokio::select! {
            biased;
            changed = runs.changed() => {
                if changed.is_err() {
                    break; // the bridge is gone
                }
                let note = runs.borrow_and_update().clone();
                run = note.run;
                if socket.send(Message::Text(note.message)).await.is_err() {
                    break;
                }
            }
            changed = snapshots.changed() => {
                if changed.is_err() {
                    break;
                }
                let snapshot = snapshots.borrow_and_update().clone();
                if let Some(text) = snapshot {
                    if socket.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
            changed = models.changed() => {
                if changed.is_err() {
                    break;
                }
                let message = models.borrow_and_update().clone();
                if let Some(text) = message {
                    if socket.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
            batch = events.recv() => match batch {
                Ok(batch) if batch.run == run => {
                    if socket.send(Message::Text(batch.message)).await.is_err() {
                        break;
                    }
                }
                Ok(_) => {} // queued before this browser heard of the run change
                Err(RecvError::Lagged(_)) => {} // this browser fell behind; those batches are gone
                Err(RecvError::Closed) => break,
            },
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
