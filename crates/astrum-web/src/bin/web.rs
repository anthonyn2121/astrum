//! `astrum-web` — stream live/replayed telemetry to the browser and serve the
//! static 3-D renderer.
//!
//! One process, two jobs, on a single TCP port:
//!  * **`GET /telemetry`** (WebSocket) — every connected browser gets one JSON
//!    [`astrum_web::FrameJson`] per telemetry sample. The upstream is a ZMQ
//!    telemetry `SUB`, so the source is either the live sim or `astrum-bag-play`
//!    — the renderer can't tell the difference.
//!  * **`GET /…`** — static files from the renderer root (`web/`).
//!
//! It's deliberately synchronous (no async runtime, matching the rest of the
//! workspace): a background thread runs the blocking ZMQ `SUB` and fans each
//! frame out to per-client channels; one thread per browser drains its channel
//! to the socket. The HTTP upgrade handshake is done by hand and the raw socket
//! handed to `tungstenite` for framing.
//!
//! Config (env):
//!  * `$ASTRUM_TELEMETRY_CONNECT` — telemetry `PUB` to subscribe to (default
//!    `tcp://127.0.0.1:5555`; `tcp://sim:5555` under compose).
//!  * `$ASTRUM_WEB_BIND` — HTTP/WS listen address (default `0.0.0.0:8080`).
//!  * `$ASTRUM_WEB_ROOT` — static renderer directory (default `web`).

use std::error::Error;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use astrum_net::{context, TelemetrySubscriber};
use astrum_web::FrameJson;
use tungstenite::handshake::derive_accept_key;
use tungstenite::protocol::{Role, WebSocket};
use tungstenite::Message;

/// Live browser clients: each holds the sending half of a bounded channel the
/// broadcast thread pushes frames into.
type Clients = Arc<Mutex<Vec<SyncSender<Arc<String>>>>>;

/// Per-client outbound queue depth. If a browser can't keep up we drop frames
/// for it (it just renders the next state) rather than growing memory.
const CLIENT_QUEUE: usize = 2048;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn main() -> Result<(), Box<dyn Error>> {
    let upstream = env_or("ASTRUM_TELEMETRY_CONNECT", "tcp://127.0.0.1:5555");
    let bind = env_or("ASTRUM_WEB_BIND", "0.0.0.0:8080");
    let root = PathBuf::from(env_or("ASTRUM_WEB_ROOT", "web"));

    let clients: Clients = Arc::new(Mutex::new(Vec::new()));

    // Background: blocking ZMQ SUB → JSON → fan out to every browser.
    {
        let clients = Arc::clone(&clients);
        let upstream = upstream.clone();
        thread::Builder::new().name("telemetry-sub".into()).spawn(move || {
            if let Err(e) = pump_telemetry(&upstream, &clients) {
                eprintln!("[astrum-web] telemetry pump exited: {e}");
            }
        })?;
    }

    let listener = TcpListener::bind(&bind)?;
    println!("[astrum-web] serving {} on http://{bind}", root.display());
    println!("[astrum-web] telemetry WebSocket at ws://{bind}/telemetry (upstream {upstream})");

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[astrum-web] accept error: {e}");
                continue;
            }
        };
        let clients = Arc::clone(&clients);
        let root = root.clone();
        thread::spawn(move || {
            if let Err(e) = handle_conn(stream, clients, &root) {
                // A dropped browser tab is routine — only note it at debug level.
                let _ = e;
            }
        });
    }
    Ok(())
}

/// Subscribe to the telemetry stream and broadcast each sample as JSON to every
/// connected browser, pruning clients whose channel has closed.
fn pump_telemetry(upstream: &str, clients: &Clients) -> Result<(), Box<dyn Error>> {
    let ctx = context();
    let sub = TelemetrySubscriber::connect(&ctx, upstream)?;
    sub.subscribe_all()?;
    loop {
        let sample = sub.recv()?;
        let json = match serde_json::to_string(&FrameJson::from_sample(&sample)) {
            Ok(j) => Arc::new(j),
            Err(_) => continue,
        };
        let mut guard = clients.lock().unwrap();
        guard.retain(|tx| match tx.try_send(Arc::clone(&json)) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => true, // slow client: skip this frame
            Err(TrySendError::Disconnected(_)) => false, // gone: prune
        });
    }
}

/// A parsed HTTP request head.
struct Head {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
}

impl Head {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// Whether this is a WebSocket upgrade request.
    fn is_websocket(&self) -> bool {
        self.header("upgrade").is_some_and(|u| u.to_ascii_lowercase().contains("websocket"))
            && self.header("sec-websocket-key").is_some()
    }
}

fn handle_conn(mut stream: TcpStream, clients: Clients, root: &Path) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(15)))?;
    let Some(head) = read_head(&mut stream)? else { return Ok(()) };

    if head.is_websocket() {
        return serve_websocket(stream, clients, &head);
    }
    if head.method != "GET" {
        return write_simple(&mut stream, 405, "Method Not Allowed", "only GET is served");
    }
    if head.path == "/telemetry" {
        return write_simple(
            &mut stream,
            426,
            "Upgrade Required",
            "connect a WebSocket to /telemetry",
        );
    }
    serve_static(&mut stream, root, &head.path)
}

/// Complete the WebSocket handshake by hand, then drain this client's channel to
/// the socket until the browser leaves.
fn serve_websocket(mut stream: TcpStream, clients: Clients, head: &Head) -> std::io::Result<()> {
    let key = head.header("sec-websocket-key").unwrap_or_default();
    let accept = derive_accept_key(key.as_bytes());
    let resp = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {accept}\r\n\r\n"
    );
    stream.write_all(resp.as_bytes())?;
    // The writer loop only writes; clear the head read-timeout so it's inert.
    stream.set_read_timeout(None)?;

    let (tx, rx) = sync_channel::<Arc<String>>(CLIENT_QUEUE);
    clients.lock().unwrap().push(tx);

    let mut ws = WebSocket::from_raw_socket(stream, Role::Server, None);
    loop {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(msg) => {
                if ws.write(Message::Text((*msg).clone())).is_err() || ws.flush().is_err() {
                    break;
                }
            }
            // Idle: ping to detect a browser that left without a clean close.
            Err(RecvTimeoutError::Timeout) => {
                if ws.write(Message::Ping(Vec::new())).is_err() || ws.flush().is_err() {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

/// Serve a file from `root`. Rejects any path that escapes the root.
fn serve_static(stream: &mut TcpStream, root: &Path, req_path: &str) -> std::io::Result<()> {
    let rel = req_path.split(['?', '#']).next().unwrap_or("/").trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };

    // Only allow plain forward path components — no `..`, no absolute/root/prefix.
    let safe = Path::new(rel).components().all(|c| matches!(c, Component::Normal(_)));
    if !safe {
        return write_simple(stream, 403, "Forbidden", "bad path");
    }

    let full = root.join(rel);
    match std::fs::read(&full) {
        Ok(bytes) => {
            let ctype = content_type(&full);
            let header = format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: {ctype}\r\n\
                 Content-Length: {}\r\n\
                 Cache-Control: no-cache\r\n\
                 Connection: close\r\n\r\n",
                bytes.len()
            );
            stream.write_all(header.as_bytes())?;
            stream.write_all(&bytes)?;
            Ok(())
        }
        Err(_) => write_simple(stream, 404, "Not Found", &format!("no such file: /{rel}")),
    }
}

fn write_simple(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    body: &str,
) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {reason}\r\n\
         Content-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())
}

/// Guess a content type from a file extension (only the ones the renderer uses).
fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}

/// Read and parse the HTTP request head (up to the blank line). `Ok(None)` on a
/// closed/oversized/timed-out connection.
fn read_head(stream: &mut TcpStream) -> std::io::Result<Option<Head>> {
    const MAX: usize = 16 * 1024;
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    loop {
        let n = match stream.read(&mut chunk) {
            Ok(0) => return Ok(None),
            Ok(n) => n,
            Err(_) => return Ok(None), // timeout / reset — nothing to serve
        };
        buf.extend_from_slice(&chunk[..n]);
        if let Some(end) = find_head_end(&buf) {
            return Ok(Some(parse_head(&buf[..end])));
        }
        if buf.len() > MAX {
            return Ok(None);
        }
    }
}

/// Index just past the CRLFCRLF that ends the head, if present.
fn find_head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

fn parse_head(bytes: &[u8]) -> Head {
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let headers = lines
        .filter(|l| !l.is_empty())
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    Head { method, path, headers }
}
