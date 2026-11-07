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
//!   `astrum/command/<id>`, stamping a random non-zero `epoch` (its instance
//!   id: the sim lets one epoch own an aircraft) and a strictly increasing
//!   `seq`; the sim's [`CommandReceiver`] drains them each tick and rejects any
//!   whose payload `target_id` disagrees with the key.
//! * **Events** — the sim publishes each [`Event`] (hit, kill, mode change, ...)
//!   on `astrum/events` ([`EventPublisher`]); [`EventReceiver`] queues them.
//! * **Tick end** — after a tick's telemetry and events the sim publishes one
//!   [`TickEnd`] on `astrum/tick` ([`TickPublisher`]); [`TickReceiver`] keeps
//!   the newest.
//! * **Scenario** — the sim answers `get astrum/scenario` with its scenario TOML
//!   ([`ScenarioServer`] / [`fetch_scenario`]), so recorders can embed it.
//! * **Presence** — a controller announces itself with a liveliness token
//!   ([`PresenceToken`]); the sim's [`PresenceWatcher`] sees it come and go.
//! * **Control and status** — the sim's [`ControlServer`] answers pause /
//!   resume / step / speed requests ([`send_control`]) and `astrum/status`
//!   ([`fetch_status`]).
//!
//! Topology is a star: the sim (or a bag replay) listens as a Zenoh router and
//! every other process connects to it as a connect-only peer, so the hub routes
//! between clients (a recorder sees the controllers' commands, a follower sees
//! the leader's telemetry).
//!
//! Per tick on which any telemetry is due the sim publishes, in this order:
//! every due telemetry sample (aircraft id ascending), then the tick's events,
//! then one [`TickEnd`]. So a subscriber that has received `TickEnd` N has
//! received every telemetry sample and event of tick N that reached it, and can
//! act on a whole, tick-aligned fleet.
//!
//! QoS ([`qos`]): telemetry and commands carry latest-value state, so a
//! congested link drops the older sample (`CongestionControl::Drop`), samples
//! are not batched (`express`) and are marked `InteractiveHigh`. Events and
//! tick ends must not be lost, so they block rather than drop and are marked
//! `Data`. What the priority buys depends on the link's transport. The sim's
//! router offers Zenoh's low-latency transport, which the C++ and Python
//! controller clients ask for (every lockstep round trip counts): it has ONE
//! lane per link, no batching and no fragmentation, so priorities reorder
//! nothing — messages arrive in the order the sim sent them — and every message
//! must fit in one transport frame. Clients that don't ask (the recorder, the
//! web bridge) get the default transport: one lane per priority (a higher one
//! is sent first) and fragmentation for large replies. Either way a link
//! delivers the messages of one priority in send order, which is what the
//! per-tick order above relies on.
//!
//! Every call uses Zenoh's synchronous `.wait()` resolver; Zenoh runs its own
//! runtime internally, so these wrappers drop into synchronous code as-is.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use astrum_proto::{
    decode_command, decode_event, decode_telemetry, decode_tick_end, encode_command,
    encode_telemetry, encode_tick_end,
};
pub use astrum_proto::{Command, Control, Event, EventKind, Telemetry, TickEnd};
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
    /// Every event the sim reports.
    pub const EVENTS: &str = "astrum/events";
    /// The end of each tick's publications ([`TickEnd`](crate::TickEnd)).
    pub const TICK: &str = "astrum/tick";
    /// `astrum/alive/controller/<id>`: liveliness token of the controller
    /// flying aircraft `<id>`.
    pub const ALIVE_PREFIX: &str = "astrum/alive/controller";
    /// Every controller's liveliness token.
    pub const ALIVE_ALL: &str = "astrum/alive/controller/*";
    /// `astrum/control/<pause|resume|step|speed>`: run control (queryable).
    pub const CONTROL_PREFIX: &str = "astrum/control";
    /// Every control request.
    pub const CONTROL_ALL: &str = "astrum/control/*";
    /// The sim's run status as JSON (queryable).
    pub const STATUS: &str = "astrum/status";

    pub fn telemetry(id: u32) -> String {
        format!("{TELEMETRY_PREFIX}/{id}")
    }

    pub fn command(id: u32) -> String {
        format!("{COMMAND_PREFIX}/{id}")
    }

    pub fn alive(id: u32) -> String {
        format!("{ALIVE_PREFIX}/{id}")
    }

    /// The id in `<prefix>/<id>`, or `None` if `key` is not of that form.
    pub fn id_of(key: &str, prefix: &str) -> Option<u32> {
        key.strip_prefix(prefix)?.strip_prefix('/')?.parse().ok()
    }
}

/// The QoS of every publication.
pub mod qos {
    use zenoh::qos::{CongestionControl, Priority};

    /// Latest-value state: drop the older sample rather than block the sender.
    pub const CONGESTION: CongestionControl = CongestionControl::Drop;
    /// Above bulk data, below Zenoh's own control traffic.
    pub const PRIORITY: Priority = Priority::InteractiveHigh;
    /// Send each sample immediately instead of batching.
    pub const EXPRESS: bool = true;
    /// Events are one-off: block rather than drop one.
    pub const EVENT_CONGESTION: CongestionControl = CongestionControl::Block;
    /// Marked below telemetry and commands.
    pub const EVENT_PRIORITY: Priority = Priority::Data;
    /// A lost tick end would stall a tick-aligned reader: block rather than drop.
    pub const TICK_CONGESTION: CongestionControl = CongestionControl::Block;
    /// The same priority as events, so it follows the tick's events on any link.
    pub const TICK_PRIORITY: Priority = Priority::Data;
    /// Sent at once: readers wait on it.
    pub const TICK_EXPRESS: bool = true;
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
    /// The sim refused a control request (the reason it gave).
    Refused(String),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::Timeout => write!(f, "receive timed out"),
            NetError::Disconnected => write!(f, "subscription channel disconnected"),
            NetError::Zenoh(e) => write!(f, "zenoh error: {e}"),
            NetError::BadScenario(e) => write!(f, "scenario reply is not UTF-8: {e}"),
            NetError::Refused(why) => write!(f, "refused: {why}"),
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
            // Low-latency transport (no batching or QoS lanes) to every client
            // that asks for it — the controllers do; measured ~14 % more lockstep
            // throughput. Clients that don't (recorder, web) get the default.
            ("transport/unicast/lowlatency", "true".into()),
            ("transport/unicast/qos/enabled", "false".into()),
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
/// with this sender's random non-zero `epoch` and a strictly increasing `seq`.
pub struct CommandSender {
    target_id: u32,
    epoch: u64,
    seq: u64,
    publisher: Publisher<'static>,
}

/// A random non-zero epoch (std's per-process random hash keys, mixed with the
/// clock so two senders in one process differ too).
fn random_epoch() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    loop {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        h.write_u32(std::process::id());
        let epoch = h.finish();
        if epoch != 0 {
            return epoch;
        }
    }
}

impl CommandSender {
    pub fn new(session: &Session, target_id: u32) -> Result<Self> {
        let publisher = session.declare_publisher(keys::command(target_id))?;
        Ok(Self { target_id, epoch: random_epoch(), seq: 0, publisher })
    }

    /// This sender's epoch (non-zero), stamped on every command it sends.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Send `control` computed from telemetry tick `telemetry_tick`. Returns the
    /// sequence number used.
    pub fn send(&mut self, control: Control, trigger: bool, telemetry_tick: u64) -> Result<u64> {
        self.seq += 1;
        let cmd = Command {
            target_id: self.target_id,
            seq: self.seq,
            telemetry_tick,
            trigger,
            epoch: self.epoch,
            control,
        };
        self.publisher.put(encode_command(&cmd)).wait().map_err(zerr)?;
        Ok(self.seq)
    }
}

// --- Events ---------------------------------------------------------------------

/// Sim-side event publisher on `astrum/events`. Events are one-off (a dropped one
/// is lost for good, unlike a telemetry sample the next tick replaces), so a
/// congested link blocks instead of dropping; Zenoh's default reliability over
/// TCP delivers the rest.
pub struct EventPublisher {
    publisher: Publisher<'static>,
}

impl EventPublisher {
    pub fn new(session: &Session) -> Result<Self> {
        let publisher = session
            .z
            .declare_publisher(keys::EVENTS)
            .congestion_control(qos::EVENT_CONGESTION)
            .priority(qos::EVENT_PRIORITY)
            .wait()
            .map_err(zerr)?;
        Ok(Self { publisher })
    }

    /// Publish an already-encoded `Event` payload.
    pub fn publish_raw(&self, payload: Vec<u8>) -> Result<()> {
        self.publisher.put(payload).wait().map_err(zerr)
    }
}

/// Subscriber-side event queue on `astrum/events`, in arrival order. Zenoh's
/// callback only decodes into an unbounded channel, so a slow reader never
/// pushes back on the transport; payloads that fail to decode (or carry a kind
/// this build doesn't know) are counted and skipped.
pub struct EventReceiver {
    rx: std::sync::mpsc::Receiver<Event>,
    skipped: Arc<std::sync::atomic::AtomicU64>,
    _sub: Subscriber<()>,
}

impl EventReceiver {
    pub fn subscribe(session: &Session) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel();
        let skipped = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let counter = Arc::clone(&skipped);
        let sub = session
            .z
            .declare_subscriber(keys::EVENTS)
            .callback(move |sample| match decode_event(&sample.payload().to_bytes()) {
                Ok(e) => {
                    let _ = tx.send(e); // fails only once the receiver is gone
                }
                Err(_) => {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            })
            .wait()
            .map_err(zerr)?;
        Ok(Self { rx, skipped, _sub: sub })
    }

    /// Every event received so far, without blocking.
    pub fn drain(&self) -> Vec<Event> {
        self.rx.try_iter().collect()
    }

    /// Block up to `timeout` for the next event.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Event> {
        self.rx.recv_timeout(timeout).map_err(|e| match e {
            std::sync::mpsc::RecvTimeoutError::Timeout => NetError::Timeout,
            std::sync::mpsc::RecvTimeoutError::Disconnected => NetError::Disconnected,
        })
    }

    /// Payloads skipped so far (undecodable, or a kind unknown to this build).
    pub fn skipped(&self) -> u64 {
        self.skipped.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// --- Tick end -------------------------------------------------------------------

/// Sim-side publisher of [`TickEnd`] on `astrum/tick`.
pub struct TickPublisher {
    publisher: Publisher<'static>,
}

impl TickPublisher {
    pub fn new(session: &Session) -> Result<Self> {
        let publisher = session
            .z
            .declare_publisher(keys::TICK)
            .congestion_control(qos::TICK_CONGESTION)
            .priority(qos::TICK_PRIORITY)
            .express(qos::TICK_EXPRESS)
            .wait()
            .map_err(zerr)?;
        Ok(Self { publisher })
    }

    /// Publish the end of tick `t.tick`'s publications.
    pub fn publish(&self, t: &TickEnd) -> Result<()> {
        self.publisher.put(encode_tick_end(t)).wait().map_err(zerr)
    }
}

#[derive(Default)]
struct TickState {
    latest: Option<TickEnd>,
    generation: u64,
    decode_errors: u64,
}

/// The newest [`TickEnd`] on `astrum/tick` (tests and tools; controllers use
/// their own SDK). Decoded on Zenoh's receive thread, overwriting the previous.
pub struct TickReceiver {
    shared: Arc<(Mutex<TickState>, Condvar)>,
    seen_generation: u64,
    _sub: Subscriber<()>,
}

impl TickReceiver {
    pub fn subscribe(session: &Session) -> Result<Self> {
        let shared: Arc<(Mutex<TickState>, Condvar)> = Arc::default();
        let sink = Arc::clone(&shared);
        let sub = session
            .z
            .declare_subscriber(keys::TICK)
            .callback(move |sample: Sample| {
                let decoded = decode_tick_end(&sample.payload().to_bytes());
                let (lock, cv) = &*sink;
                let mut st = lock.lock().unwrap_or_else(PoisonError::into_inner);
                match decoded {
                    Ok(t) => {
                        st.latest = Some(t);
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

    fn state(&self) -> std::sync::MutexGuard<'_, TickState> {
        self.shared.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The newest tick end, if any arrived.
    pub fn latest(&self) -> Option<TickEnd> {
        self.state().latest
    }

    /// Payloads that failed to decode so far.
    pub fn decode_errors(&self) -> u64 {
        self.state().decode_errors
    }

    /// Block until a tick end newer than the last `wait` arrives; returns it.
    pub fn wait(&mut self, timeout: Duration) -> Result<TickEnd> {
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
        Ok(st.latest.expect("a generation bump stores a tick end"))
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

// --- Controller presence --------------------------------------------------------

/// A controller's announcement that it flies aircraft `id`: a Zenoh liveliness
/// token on `astrum/alive/controller/<id>` that lasts as long as this value and
/// its session. A crashed process loses its connection, so the sim learns of it
/// at once instead of waiting for a lease or lockstep timeout.
pub struct PresenceToken {
    _token: zenoh::liveliness::LivelinessToken,
}

impl PresenceToken {
    pub fn declare(session: &Session, aircraft_id: u32) -> Result<Self> {
        let token =
            session.z.liveliness().declare_token(keys::alive(aircraft_id)).wait().map_err(zerr)?;
        Ok(Self { _token: token })
    }
}

/// A change in which controllers are present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    Up(u32),
    Down(u32),
}

/// Sim-side watcher of controller tokens: reports the ones already present when
/// it starts, then every change.
pub struct PresenceWatcher {
    sub: Subscriber<FifoChannelHandler<Sample>>,
}

impl PresenceWatcher {
    pub fn watch(session: &Session) -> Result<Self> {
        let sub = session
            .z
            .liveliness()
            .declare_subscriber(keys::ALIVE_ALL)
            .history(true)
            .with(zenoh::handlers::FifoChannel::new(COMMAND_QUEUE))
            .wait()
            .map_err(zerr)?;
        Ok(Self { sub })
    }

    /// Every change since the last call, in order, without blocking. Tokens on a
    /// key without a numeric id are ignored.
    pub fn drain(&self) -> Vec<Presence> {
        let mut out = Vec::new();
        while let Ok(Some(sample)) = self.sub.try_recv() {
            if let Some(id) = keys::id_of(sample.key_expr().as_str(), keys::ALIVE_PREFIX) {
                out.push(match sample.kind() {
                    zenoh::sample::SampleKind::Put => Presence::Up(id),
                    zenoh::sample::SampleKind::Delete => Presence::Down(id),
                });
            }
        }
        out
    }
}

// --- Run control ----------------------------------------------------------------

/// Slowest supported `sim.speed` other than `0` (unpaced).
pub const MIN_SPEED: f64 = 1e-3;

/// A request on `astrum/control/*`. Control decides only when ticks run (pause,
/// step, pace), never what a tick computes, so a lockstep run is unchanged by it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlRequest {
    /// `astrum/control/pause`: stop after the current tick.
    Pause,
    /// `astrum/control/resume`: run on.
    Resume,
    /// `astrum/control/step?ticks=N`: run N ticks, then pause.
    Step(u64),
    /// `astrum/control/speed?value=X`: the new `sim.speed` (0 = unpaced).
    Speed(f64),
}

impl ControlRequest {
    /// The selector (`key?parameters`) a client queries.
    pub fn selector(&self) -> String {
        match self {
            ControlRequest::Pause => format!("{}/pause", keys::CONTROL_PREFIX),
            ControlRequest::Resume => format!("{}/resume", keys::CONTROL_PREFIX),
            ControlRequest::Step(n) => format!("{}/step?ticks={n}", keys::CONTROL_PREFIX),
            ControlRequest::Speed(x) => format!("{}/speed?value={x}", keys::CONTROL_PREFIX),
        }
    }

    /// Check a request's values: `Step` needs at least one tick, `Speed` must
    /// be 0 (unpaced) or finite and at least `min_speed`.
    pub fn validate(self, min_speed: f64) -> Result<Self, String> {
        match self {
            ControlRequest::Step(0) => Err("step needs ticks >= 1, got 0".into()),
            ControlRequest::Speed(x) if !(x == 0.0 || (x.is_finite() && x >= min_speed)) => {
                Err(format!("speed must be 0 (unpaced) or >= {min_speed}, got {x}"))
            }
            valid => Ok(valid),
        }
    }

    /// Parse and check a request; `min_speed` is the slowest non-zero speed.
    fn parse(
        key: &str,
        param: impl Fn(&str) -> Option<String>,
        min_speed: f64,
    ) -> Result<Self, String> {
        let op = key.strip_prefix(keys::CONTROL_PREFIX).and_then(|k| k.strip_prefix('/'));
        let request = match op {
            Some("pause") => ControlRequest::Pause,
            Some("resume") => ControlRequest::Resume,
            Some("step") => match param("ticks").map(|t| t.parse::<u64>()) {
                Some(Ok(n)) => ControlRequest::Step(n),
                _ => return Err("step needs ?ticks=<N> with N >= 1".into()),
            },
            Some("speed") => match param("value").map(|v| v.parse::<f64>()) {
                Some(Ok(x)) => ControlRequest::Speed(x),
                _ => {
                    return Err(format!(
                        "speed needs ?value=<X> with X = 0 (unpaced) or >= {min_speed}"
                    ))
                }
            },
            _ => return Err(format!("unknown control {key} (pause, resume, step, speed)")),
        };
        request.validate(min_speed)
    }
}

/// The sim's run status, answered on `astrum/status` as JSON.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunStatus {
    pub tick: u64,
    pub sim_time_s: f64,
    /// `running`, `paused`, or `waiting` (a lockstep tick waiting on controllers).
    pub state: String,
    /// Target pace (× real time; 0 = unpaced).
    pub speed: f64,
    /// Pace actually achieved recently (× real time).
    pub achieved_speed: f64,
    pub aircraft: usize,
    /// Aircraft whose controller is present (liveliness).
    pub controllers: Vec<u32>,
    /// While waiting: the lockstep aircraft whose command is still missing.
    pub lockstep_waiting: Vec<u32>,
}

/// The sim's status as last written by the sim, read by the status queryable.
pub type StatusBoard = Arc<Mutex<RunStatus>>;

/// Sim side of the control plane: validates control requests on Zenoh's thread
/// (bad ones get an error reply), queues the good ones for the sim to apply
/// between ticks, and answers `astrum/status` from the [`StatusBoard`].
pub struct ControlServer {
    _control: Queryable<()>,
    _status: Queryable<()>,
    requests: std::sync::mpsc::Receiver<ControlRequest>,
    board: StatusBoard,
}

impl ControlServer {
    pub fn serve(session: &Session, min_speed: f64) -> Result<Self> {
        let (tx, requests) = std::sync::mpsc::channel();
        let control = session
            .z
            .declare_queryable(keys::CONTROL_ALL)
            .callback(move |query| {
                let key = query.key_expr().as_str().to_owned();
                let params = query.parameters();
                let param = |name: &str| params.get(name).map(str::to_owned);
                let _ = match ControlRequest::parse(&key, param, min_speed) {
                    Ok(req) if tx.send(req).is_ok() => query.reply(key.clone(), "ok").wait(),
                    Ok(_) => query.reply_err("the sim is shutting down").wait(),
                    Err(why) => query.reply_err(why).wait(),
                };
            })
            .wait()
            .map_err(zerr)?;
        let board = StatusBoard::default();
        let shown = Arc::clone(&board);
        let status = session
            .z
            .declare_queryable(keys::STATUS)
            .callback(move |query| {
                let json = {
                    let s = shown.lock().unwrap_or_else(PoisonError::into_inner);
                    serde_json::to_string(&*s).unwrap_or_default()
                };
                let _ = query.reply(keys::STATUS, json).wait();
            })
            .wait()
            .map_err(zerr)?;
        Ok(Self { _control: control, _status: status, requests, board })
    }

    /// Requests received since the last call, in order.
    pub fn drain(&self) -> Vec<ControlRequest> {
        self.requests.try_iter().collect()
    }

    /// Like [`drain`](Self::drain), but wait up to `timeout` for the first one.
    pub fn wait(&self, timeout: Duration) -> Vec<ControlRequest> {
        match self.requests.recv_timeout(timeout) {
            Ok(first) => std::iter::once(first).chain(self.requests.try_iter()).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// The board the sim writes its status to.
    pub fn board(&self) -> StatusBoard {
        Arc::clone(&self.board)
    }
}

/// One query, one reply: its payload, or the sim's refusal.
fn query_one(session: &Session, selector: &str, timeout: Duration) -> Result<Vec<u8>> {
    let replies = session.z.get(selector).timeout(timeout).wait().map_err(zerr)?;
    match replies.recv_timeout(timeout + Duration::from_millis(100)) {
        Ok(Some(reply)) => match reply.result() {
            Ok(sample) => Ok(sample.payload().to_bytes().into_owned()),
            Err(e) => {
                Err(NetError::Refused(String::from_utf8_lossy(&e.payload().to_bytes()).into()))
            }
        },
        _ => Err(NetError::Timeout),
    }
}

/// Ask the sim to apply `request`; an invalid request comes back as
/// [`NetError::Refused`], no sim as [`NetError::Timeout`].
pub fn send_control(session: &Session, request: ControlRequest, timeout: Duration) -> Result<()> {
    query_one(session, &request.selector(), timeout).map(drop)
}

/// The sim's current [`RunStatus`].
pub fn fetch_status(session: &Session, timeout: Duration) -> Result<RunStatus> {
    let bytes = query_one(session, keys::STATUS, timeout)?;
    serde_json::from_slice(&bytes).map_err(|e| NetError::Zenoh(format!("bad status reply: {e}")))
}
