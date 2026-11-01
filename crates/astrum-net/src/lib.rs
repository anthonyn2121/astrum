//! `astrum-net` — the ASTRUM Zenoh transport layer.
//!
//! One Zenoh session per process carries every stream, each keyed by aircraft id
//! (see [`keys`]):
//!
//! * **Telemetry** — the sim publishes each [`Telemetry`] on
//!   `astrum/telemetry/<id>` through a per-aircraft declared publisher.
//!   [`LatestTelemetry`] keeps the newest sample per aircraft for controllers and
//!   viewers: a control law must act on the freshest state, never on a queue of
//!   stale frames.
//! * **Commands** — a controller's [`CommandSender`] publishes on
//!   `astrum/command/<id>`; the sim's [`CommandReceiver`] drains them each tick
//!   and rejects any whose payload `target_id` disagrees with the key.
//! * **Scenario** — the sim answers `get astrum/scenario` with its scenario TOML
//!   ([`ScenarioServer`] / [`fetch_scenario`]), so recorders can embed it.
//!
//! Topology is a star: the sim (or a bag replay) listens as a Zenoh router and
//! every other process connects to it as a connect-only peer, so the hub routes
//! between clients (a recorder sees the controllers' commands, a follower sees
//! the leader's telemetry).
//!
//! QoS ([`qos`]) is the same for both streams: they carry latest-value state, so a
//! congested link drops the older sample (`CongestionControl::Drop`), samples are
//! not batched (`express`), and they ride above bulk data (`InteractiveHigh`).
//!
//! Every call uses Zenoh's synchronous `.wait()` resolver; Zenoh runs its own
//! runtime internally, so these wrappers drop into synchronous code as-is.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use astrum_proto::{decode_command, decode_telemetry, encode_command, encode_telemetry};
pub use astrum_proto::{Command, Control, Telemetry};
use zenoh::handlers::FifoChannelHandler;
use zenoh::pubsub::{Publisher, Subscriber};
use zenoh::query::Queryable;
use zenoh::sample::Sample;
use zenoh::Wait;

// Re-export so downstream crates share one zenoh version / type set.
pub use zenoh;

/// Key expressions.
pub mod keys {
    /// `astrum/telemetry/<id>`
    pub const TELEMETRY_PREFIX: &str = "astrum/telemetry";
    /// `astrum/command/<id>`
    pub const COMMAND_PREFIX: &str = "astrum/command";
    /// Every aircraft's telemetry.
    pub const TELEMETRY_ALL: &str = "astrum/telemetry/*";
    /// Every aircraft's commands.
    pub const COMMAND_ALL: &str = "astrum/command/*";
    /// The sim's scenario TOML (queryable).
    pub const SCENARIO: &str = "astrum/scenario";

    pub fn telemetry(id: u32) -> String {
        format!("{TELEMETRY_PREFIX}/{id}")
    }

    pub fn command(id: u32) -> String {
        format!("{COMMAND_PREFIX}/{id}")
    }

    /// The id in `<prefix>/<id>`, or `None` if `key` is not of that form.
    pub fn id_of(key: &str, prefix: &str) -> Option<u32> {
        key.strip_prefix(prefix)?.strip_prefix('/')?.parse().ok()
    }
}

/// The QoS every telemetry and command publication uses.
pub mod qos {
    use zenoh::qos::{CongestionControl, Priority};

    /// Latest-value state: drop the older sample rather than block the sender.
    pub const CONGESTION: CongestionControl = CongestionControl::Drop;
    /// Above bulk data, below Zenoh's own control traffic.
    pub const PRIORITY: Priority = Priority::InteractiveHigh;
    /// Send each sample immediately instead of batching.
    pub const EXPRESS: bool = true;
}

/// Default endpoint the sim (and bag replay) LISTENS on. Zenoh endpoints use
/// `proto/host:port` (a single `/`, not `://`).
pub const DEFAULT_LISTEN_ENDPOINT: &str = "tcp/0.0.0.0:7447";
/// Default endpoint a client CONNECTS to (the sim on localhost).
pub const DEFAULT_CONNECT_ENDPOINT: &str = "tcp/127.0.0.1:7447";

/// Transport errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetError {
    /// A receive timed out.
    Timeout,
    /// The subscription's channel closed (the session went away).
    Disconnected,
    /// An underlying Zenoh session / declaration / publication error.
    Zenoh(String),
    /// A scenario reply was not UTF-8.
    BadScenario(String),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::Timeout => write!(f, "receive timed out"),
            NetError::Disconnected => write!(f, "subscription channel disconnected"),
            NetError::Zenoh(e) => write!(f, "zenoh error: {e}"),
            NetError::BadScenario(e) => write!(f, "scenario reply is not UTF-8: {e}"),
        }
    }
}

impl std::error::Error for NetError {}

/// Transport result alias.
pub type Result<T, E = NetError> = std::result::Result<T, E>;

fn zerr<E: fmt::Display>(e: E) -> NetError {
    NetError::Zenoh(e.to_string())
}

/// A Zenoh session. Cheap to clone (an `Arc` inside).
#[derive(Clone)]
pub struct Session {
    z: zenoh::Session,
}

impl Session {
    /// Open a session that LISTENS on `endpoint` (the sim / bag-replay role) as a
    /// Zenoh **router**: every other process connects only to this hub, and peers
    /// do not forward for each other, so the hub must route — otherwise one
    /// client's publications (a controller's commands) would never reach another
    /// client (the recorder). Multicast scouting is off: Docker bridge networks
    /// don't pass multicast, so clients connect by explicit endpoint.
    pub fn listen(endpoint: &str) -> Result<Self> {
        Self::open(&[
            ("scouting/multicast/enabled", "false".into()),
            ("mode", r#""router""#.into()),
            ("listen/endpoints", format!(r#"["{endpoint}"]"#)),
        ])
    }

    /// Open a session that CONNECTS to the hub at `endpoint` (controller /
    /// recorder / web) as a Zenoh **client**: the hub routes everything for it,
    /// including other clients' publications. Waits up to `timeout` for the hub
    /// (a sim still loading its aircraft), then fails; once connected, Zenoh
    /// reconnects by itself if the hub restarts.
    pub fn connect(endpoint: &str, timeout: Duration) -> Result<Self> {
        let timeout_ms = i64::try_from(timeout.as_millis()).unwrap_or(i64::MAX);
        Self::open(&[
            ("scouting/multicast/enabled", "false".into()),
            ("mode", r#""client""#.into()),
            ("connect/endpoints", format!(r#"["{endpoint}"]"#)),
            ("connect/timeout_ms", timeout_ms.to_string()),
        ])
    }

    /// An isolated in-process session (no endpoints) whose publishers deliver to
    /// its own subscribers — for tests.
    pub fn in_process() -> Result<Self> {
        Self::open(&[("scouting/multicast/enabled", "false".into()), ("mode", r#""peer""#.into())])
    }

    fn open(patches: &[(&str, String)]) -> Result<Self> {
        let mut config = zenoh::Config::default();
        for (key, value) in patches {
            config.insert_json5(key, value).map_err(zerr)?;
        }
        let z = zenoh::open(config).wait().map_err(zerr)?;
        Ok(Self { z })
    }

    /// The underlying Zenoh session (raw recording / replay, advanced use).
    pub fn raw(&self) -> &zenoh::Session {
        &self.z
    }

    /// Declare a publisher on `key` with the ASTRUM [`qos`].
    pub fn declare_publisher(&self, key: String) -> Result<Publisher<'static>> {
        self.z
            .declare_publisher(key)
            .congestion_control(qos::CONGESTION)
            .priority(qos::PRIORITY)
            .express(qos::EXPRESS)
            .wait()
            .map_err(zerr)
    }
}

// --- Telemetry ----------------------------------------------------------------

/// Sim-side telemetry broadcaster: one declared publisher per aircraft, created
/// on first use, so each key expression is resolved and declared once.
pub struct TelemetryPublisher {
    session: Session,
    publishers: HashMap<u32, Publisher<'static>>,
}

impl TelemetryPublisher {
    pub fn new(session: &Session) -> Self {
        Self { session: session.clone(), publishers: HashMap::new() }
    }

    /// Publish one sample on `astrum/telemetry/<t.aircraft_id>`.
    pub fn publish(&mut self, t: &Telemetry) -> Result<()> {
        self.publish_raw(t.aircraft_id, encode_telemetry(t))
    }

    /// Publish an already-encoded `Telemetry` payload (bag replay).
    pub fn publish_raw(&mut self, aircraft_id: u32, payload: Vec<u8>) -> Result<()> {
        let publisher = match self.publishers.entry(aircraft_id) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(self.session.declare_publisher(keys::telemetry(aircraft_id))?)
            }
        };
        publisher.put(payload).wait().map_err(zerr)
    }
}

/// Which telemetry a [`LatestTelemetry`] tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    Aircraft(u32),
    Fleet,
}

#[derive(Default)]
struct LatestState {
    latest: BTreeMap<u32, Telemetry>,
    generation: u64,
    decode_errors: u64,
}

type Shared = Arc<(Mutex<LatestState>, Condvar)>;

/// The newest telemetry sample per aircraft. Samples are decoded on Zenoh's
/// receive thread and overwrite the previous one for that aircraft, so a slow
/// consumer always sees current state and never blocks the transport.
pub struct LatestTelemetry {
    shared: Shared,
    seen_generation: u64,
    _sub: Subscriber<()>,
}

impl LatestTelemetry {
    pub fn subscribe(session: &Session, selection: Selection) -> Result<Self> {
        let key = match selection {
            Selection::Aircraft(id) => keys::telemetry(id),
            Selection::Fleet => keys::TELEMETRY_ALL.to_owned(),
        };
        let shared: Shared = Arc::default();
        let sink = Arc::clone(&shared);
        let sub = session
            .z
            .declare_subscriber(key)
            .callback(move |sample: Sample| {
                let decoded = decode_telemetry(&sample.payload().to_bytes());
                let (lock, cv) = &*sink;
                let mut st = lock.lock().unwrap_or_else(PoisonError::into_inner);
                match decoded {
                    Ok(t) => {
                        st.latest.insert(t.aircraft_id, t);
                        st.generation += 1;
                    }
                    Err(_) => st.decode_errors += 1,
                }
                drop(st);
                cv.notify_all();
            })
            .wait()
            .map_err(zerr)?;
        Ok(Self { shared, seen_generation: 0, _sub: sub })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, LatestState> {
        self.shared.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The newest sample for `aircraft_id`, if any arrived.
    pub fn latest(&self, aircraft_id: u32) -> Option<Telemetry> {
        self.state().latest.get(&aircraft_id).copied()
    }

    /// The newest sample of every aircraft seen, ordered by id.
    pub fn snapshot(&self) -> Vec<Telemetry> {
        self.state().latest.values().copied().collect()
    }

    /// Payloads that failed to decode so far.
    pub fn decode_errors(&self) -> u64 {
        self.state().decode_errors
    }

    /// Block until at least one sample newer than the last `wait` arrives.
    pub fn wait(&mut self, timeout: Duration) -> Result<()> {
        let (lock, cv) = &*self.shared;
        let deadline = Instant::now() + timeout;
        let mut st = lock.lock().unwrap_or_else(PoisonError::into_inner);
        while st.generation == self.seen_generation {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(NetError::Timeout);
            }
            st = cv.wait_timeout(st, left).unwrap_or_else(PoisonError::into_inner).0;
        }
        self.seen_generation = st.generation;
        Ok(())
    }
}

// --- Commands -----------------------------------------------------------------

/// Why the sim refused a command payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandReject {
    pub key: String,
    pub reason: String,
}

impl fmt::Display for CommandReject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "command on {} rejected: {}", self.key, self.reason)
    }
}

/// Queue depth of the sim's command intake: far above one tick's worth of
/// commands for any fleet, so the intake never pushes back on Zenoh in practice.
const COMMAND_QUEUE: usize = 4096;

/// Sim-side command intake on `astrum/command/*`.
pub struct CommandReceiver {
    sub: Subscriber<FifoChannelHandler<Sample>>,
}

impl CommandReceiver {
    pub fn bind(session: &Session) -> Result<Self> {
        let sub = session
            .z
            .declare_subscriber(keys::COMMAND_ALL)
            .with(zenoh::handlers::FifoChannel::new(COMMAND_QUEUE))
            .wait()
            .map_err(zerr)?;
        Ok(Self { sub })
    }

    fn validate(sample: &Sample) -> Result<Command, CommandReject> {
        let key = sample.key_expr().as_str();
        let reject = |reason: String| CommandReject { key: key.to_owned(), reason };
        let key_id = keys::id_of(key, keys::COMMAND_PREFIX)
            .ok_or_else(|| reject("key is not astrum/command/<id>".into()))?;
        let cmd = decode_command(&sample.payload().to_bytes())
            .map_err(|e| reject(format!("decode failed: {e}")))?;
        if cmd.target_id != key_id {
            return Err(reject(format!(
                "payload targets {} but key names {key_id}",
                cmd.target_id
            )));
        }
        Ok(cmd)
    }

    /// Every command queued so far, in arrival order, without blocking.
    pub fn drain(&self) -> Vec<Result<Command, CommandReject>> {
        let mut out = Vec::new();
        while let Ok(Some(sample)) = self.sub.try_recv() {
            out.push(Self::validate(&sample));
        }
        out
    }

    /// Block up to `timeout` for the next command.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Result<Command, CommandReject>> {
        match self.sub.recv_timeout(timeout) {
            Ok(Some(sample)) => Ok(Self::validate(&sample)),
            Ok(None) => Err(NetError::Timeout),
            Err(_) => Err(NetError::Disconnected),
        }
    }
}

/// Controller-side command publisher for one aircraft. Stamps every command
/// with a strictly increasing `seq`.
pub struct CommandSender {
    target_id: u32,
    seq: u64,
    publisher: Publisher<'static>,
}

impl CommandSender {
    pub fn new(session: &Session, target_id: u32) -> Result<Self> {
        let publisher = session.declare_publisher(keys::command(target_id))?;
        Ok(Self { target_id, seq: 0, publisher })
    }

    /// Send `control` computed from telemetry tick `telemetry_tick`. Returns the
    /// sequence number used.
    pub fn send(&mut self, control: Control, trigger: bool, telemetry_tick: u64) -> Result<u64> {
        self.seq += 1;
        let cmd =
            Command { target_id: self.target_id, seq: self.seq, telemetry_tick, trigger, control };
        self.publisher.put(encode_command(&cmd)).wait().map_err(zerr)?;
        Ok(self.seq)
    }
}

// --- Scenario -----------------------------------------------------------------

/// Answers `get astrum/scenario` with the sim's scenario TOML for as long as it
/// is alive.
pub struct ScenarioServer {
    _queryable: Queryable<()>,
}

impl ScenarioServer {
    pub fn serve(session: &Session, scenario_toml: String) -> Result<Self> {
        let queryable = session
            .z
            .declare_queryable(keys::SCENARIO)
            .callback(move |query| {
                // A failed reply (the querier went away) affects only that querier.
                let _ = query.reply(keys::SCENARIO, scenario_toml.clone()).wait();
            })
            .wait()
            .map_err(zerr)?;
        Ok(Self { _queryable: queryable })
    }
}

/// How often [`fetch_scenario`] re-asks while nobody has answered.
const SCENARIO_RETRY: Duration = Duration::from_millis(100);

/// Ask the sim for its scenario TOML, re-asking until `timeout` elapses (a
/// query sent before the sim declared its queryable gets no reply at all).
/// `Ok(None)` if nobody answered in time.
pub fn fetch_scenario(session: &Session, timeout: Duration) -> Result<Option<String>> {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Ok(None);
        }
        let attempt = left.min(SCENARIO_RETRY);
        let started = Instant::now();
        let replies = session.z.get(keys::SCENARIO).timeout(attempt).wait().map_err(zerr)?;
        while let Ok(reply) = replies.recv() {
            if let Ok(sample) = reply.result() {
                let text = String::from_utf8(sample.payload().to_bytes().into_owned())
                    .map_err(|e| NetError::BadScenario(e.to_string()))?;
                return Ok(Some(text));
            }
        }
        // No queryable yet: the query finished early; wait out the attempt.
        std::thread::sleep(attempt.saturating_sub(started.elapsed()));
    }
}
