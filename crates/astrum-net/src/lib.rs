//! `astrum-net` — the ASTRUM Zenoh transport layer.
//!
//! Two channels, both carrying [`astrum_proto`] payloads on the wire, multiplexed
//! over a single Zenoh session (one TCP endpoint instead of ZMQ's two ports):
//!
//! * **Telemetry** — the sim publishes each [`TelemetrySample`] on the key
//!   `astrum/telemetry/<aircraft-id>`. Subscribers take one aircraft with
//!   [`TelemetrySubscriber::subscribe`] (`astrum/telemetry/<id>`) or the whole
//!   fleet with [`TelemetrySubscriber::subscribe_all`] (`astrum/telemetry/*`).
//!   The aircraft id lives in the key, so no fixed-width topic frame is needed —
//!   key expressions match exactly (`.../1` never matches `.../10`).
//! * **Commands** — controllers publish each [`CommandMsg`] on
//!   `astrum/command/<target-id>`; the sim subscribes `astrum/command/**` and
//!   drains them non-blockingly each tick (fan-in from every controller).
//! * **Racing task state** — the drone-racing environment adds a third key space,
//!   `astrum/race/course/<drone-id>` and `astrum/race/state/<drone-id>`
//!   ([`RacePublisher`] / [`RaceSubscriber`]), for the gates and the run's
//!   progress. It flies on the two channels above unchanged: a quadrotor's
//!   telemetry is a [`TelemetrySample`] and its controls are a [`CommandMsg`].
//!   These two are **state** channels, so they use a ring of one rather than a FIFO
//!   queue: reads are latest-wins, and an idle reader can never block the sim.
//!
//! The bag recorder is just another subscriber, and replay just another
//! publisher, so replay feeds any subscriber byte-for-byte identically to the
//! live sim.
//!
//! Zenoh embeds its own async runtime; every call here uses the **synchronous**
//! `.wait()` resolver and the subscriber's blocking/non-blocking channel
//! (`recv` / `recv_timeout` / `try_recv`), so these wrappers drop straight into
//! Bevy's synchronous systems with no runtime for us to drive. Unlike the old
//! `zmq::Socket`, a Zenoh [`Session`] is `Send + Sync`, but the sim still holds
//! the wrappers as Bevy `NonSend` resources (single-threaded under
//! `ScheduleRunnerPlugin`) — no change to the bridge systems was needed.

use std::fmt;
use std::time::Duration;

use astrum_proto::{
    decode_command, decode_race_course, decode_race_state, decode_telemetry, encode_command,
    encode_race_course, encode_race_state, encode_telemetry, CommandMsg, RaceCourseMsg,
    RaceStateMsg, TelemetrySample,
};
use zenoh::qos::CongestionControl;
use zenoh::Wait;

// Re-export so downstream crates share one zenoh version / type set.
pub use zenoh;

/// Key-expression prefix for telemetry (`astrum/telemetry/<id>`).
const TELEMETRY_PREFIX: &str = "astrum/telemetry";
/// Key-expression prefix for commands (`astrum/command/<target-id>`).
const COMMAND_PREFIX: &str = "astrum/command";
/// Key-expression prefix for a racing course (`astrum/race/course/<drone-id>`).
const RACE_COURSE_PREFIX: &str = "astrum/race/course";
/// Key-expression prefix for racing progress (`astrum/race/state/<drone-id>`).
const RACE_STATE_PREFIX: &str = "astrum/race/state";

/// Default endpoint the sim (and bag replay) LISTENS on — clients connect here.
/// Zenoh endpoints use `proto/host:port` (note: `/`, not `://`).
pub const DEFAULT_LISTEN_ENDPOINT: &str = "tcp/0.0.0.0:7447";
/// Default endpoint a client CONNECTS to (the sim on localhost).
pub const DEFAULT_CONNECT_ENDPOINT: &str = "tcp/127.0.0.1:7447";

/// The telemetry key for one aircraft.
#[inline]
fn telemetry_key(aircraft_id: u32) -> String {
    format!("{TELEMETRY_PREFIX}/{aircraft_id}")
}

/// The command key targeting one aircraft.
#[inline]
fn command_key(target_id: u32) -> String {
    format!("{COMMAND_PREFIX}/{target_id}")
}

/// The course key for one racing drone.
#[inline]
fn race_course_key(drone_id: u32) -> String {
    format!("{RACE_COURSE_PREFIX}/{drone_id}")
}

/// The progress key for one racing drone.
#[inline]
fn race_state_key(drone_id: u32) -> String {
    format!("{RACE_STATE_PREFIX}/{drone_id}")
}

/// The concrete subscriber type: a Zenoh subscriber with the default FIFO channel
/// handler, whose `recv` / `recv_timeout` / `try_recv` we use synchronously.
type FifoSubscriber =
    zenoh::pubsub::Subscriber<zenoh::handlers::FifoChannelHandler<zenoh::sample::Sample>>;

/// Transport errors surfaced to callers.
#[derive(Debug)]
pub enum NetError {
    /// A configured receive timeout elapsed with no sample (ZMQ `EAGAIN` analogue).
    Timeout,
    /// The subscriber's channel disconnected (the session was closed).
    Disconnected,
    /// `recv`/`try_recv` was called before `subscribe`/`subscribe_all`.
    NotSubscribed,
    /// A received payload failed to decode.
    Decode,
    /// An underlying Zenoh session / declaration / publication error.
    Zenoh(String),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::Timeout => write!(f, "receive timed out"),
            NetError::Disconnected => write!(f, "subscription channel disconnected"),
            NetError::NotSubscribed => write!(f, "not subscribed (call subscribe/subscribe_all)"),
            NetError::Decode => write!(f, "payload decode failed"),
            NetError::Zenoh(e) => write!(f, "zenoh error: {e}"),
        }
    }
}

impl std::error::Error for NetError {}

/// Transport result alias.
pub type Result<T> = std::result::Result<T, NetError>;

fn zerr<E: fmt::Display>(e: E) -> NetError {
    NetError::Zenoh(e.to_string())
}

/// A Zenoh session — the transport handle every wrapper is built from. Cheap to
/// clone (an `Arc` inside), so it can be shared across the telemetry publisher
/// and command receiver in one process.
#[derive(Clone)]
pub struct Session {
    z: zenoh::Session,
}

impl Session {
    /// Open a session that LISTENS on `endpoint` (the sim / bag-replay role):
    /// mode `peer`, multicast scouting disabled (Docker bridge networks don't
    /// pass multicast — peers connect by explicit endpoint).
    pub fn listen(endpoint: &str) -> Result<Self> {
        Self::open(&[
            ("scouting/multicast/enabled", "false".into()),
            ("mode", r#""peer""#.into()),
            ("listen/endpoints", format!(r#"["{endpoint}"]"#)),
        ])
    }

    /// Open a session that CONNECTS to `endpoint` (the controller / recorder /
    /// web role): mode `peer`, connect-out only (no listener), multicast off.
    ///
    /// Peer mode is what makes the open non-fatal when the sim isn't listening
    /// yet: a `peer` `open()` returns immediately and Zenoh retries the outbound
    /// connection in the background (infinite retry per `connect/retry`), so a
    /// process started alongside the sim just attaches once the sim comes up —
    /// mirroring ZMQ's lazy, tolerant `connect`. A `client` instead *aborts* its
    /// `open()` if the endpoint isn't reachable the instant it starts (`client`
    /// defaults to `connect/timeout_ms: 0`), which loses the startup race with
    /// the sim's ~10s JSBSim load. `listen/endpoints: []` keeps this peer purely
    /// connect-out (the bidirectional link to the sim carries pub/sub both ways),
    /// so it behaves like the old client role without needing to accept inbound.
    pub fn connect(endpoint: &str) -> Result<Self> {
        Self::open(&[
            ("scouting/multicast/enabled", "false".into()),
            ("mode", r#""peer""#.into()),
            ("connect/endpoints", format!(r#"["{endpoint}"]"#)),
            ("listen/endpoints", "[]".into()),
        ])
    }

    /// Open an isolated in-process session (no endpoints, multicast off) whose own
    /// publishers deliver to its own subscribers — used by the integration tests
    /// in place of ZMQ's `inproc://` shared context.
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

    /// Borrow the underlying Zenoh session (for advanced use / downstream crates).
    pub fn raw(&self) -> &zenoh::Session {
        &self.z
    }
}

/// Sim-side telemetry broadcaster. Publishes on `astrum/telemetry/<id>` with
/// best-effort delivery (a dropped frame is non-fatal — the subscriber just
/// renders the next state), matching ZMQ `PUB` semantics.
pub struct TelemetryPublisher {
    session: Session,
}

impl TelemetryPublisher {
    /// Build a publisher on `session`.
    pub fn new(session: &Session) -> Self {
        Self { session: session.clone() }
    }

    /// Publish one sample on `astrum/telemetry/<sample.aircraft_id>`.
    pub fn publish(&self, sample: &TelemetrySample) -> Result<()> {
        let payload = encode_telemetry(sample);
        self.session
            .z
            .put(telemetry_key(sample.aircraft_id), payload)
            .congestion_control(CongestionControl::Drop)
            .wait()
            .map_err(zerr)
    }
}

/// Subscriber-side telemetry reader — used by the bag recorder, controllers, and
/// the replay renderer. Construction is two-step to mirror the old ZMQ `SUB`:
/// [`connect`](Self::connect) opens the reader, then [`subscribe`](Self::subscribe)
/// / [`subscribe_all`](Self::subscribe_all) declares the actual Zenoh subscriber.
pub struct TelemetrySubscriber {
    session: Session,
    sub: Option<FifoSubscriber>,
    timeout: Option<Duration>,
}

impl TelemetrySubscriber {
    /// Prepare a reader on `session`. Nothing is delivered until you call
    /// [`subscribe`](Self::subscribe) or [`subscribe_all`](Self::subscribe_all).
    pub fn connect(session: &Session) -> Result<Self> {
        Ok(Self { session: session.clone(), sub: None, timeout: None })
    }

    /// Subscribe to a single aircraft's telemetry (`astrum/telemetry/<id>`).
    pub fn subscribe(&mut self, aircraft_id: u32) -> Result<()> {
        self.declare(telemetry_key(aircraft_id))
    }

    /// Subscribe to every aircraft (`astrum/telemetry/*`).
    pub fn subscribe_all(&mut self) -> Result<()> {
        self.declare(format!("{TELEMETRY_PREFIX}/*"))
    }

    fn declare(&mut self, key: String) -> Result<()> {
        let sub = self.session.z.declare_subscriber(key).wait().map_err(zerr)?;
        self.sub = Some(sub);
        Ok(())
    }

    /// Set a receive timeout in milliseconds; a negative value blocks forever
    /// (the default). With a timeout set, [`recv`](Self::recv) returns
    /// [`NetError::Timeout`] when it elapses.
    pub fn set_recv_timeout_ms(&mut self, ms: i32) {
        self.timeout = if ms < 0 { None } else { Some(Duration::from_millis(ms as u64)) };
    }

    /// Block for the next sample (honouring any configured timeout).
    pub fn recv(&self) -> Result<TelemetrySample> {
        let sub = self.sub.as_ref().ok_or(NetError::NotSubscribed)?;
        let sample = match self.timeout {
            Some(t) => match sub.recv_timeout(t).map_err(|_| NetError::Disconnected)? {
                Some(s) => s,
                None => return Err(NetError::Timeout),
            },
            None => sub.recv().map_err(|_| NetError::Disconnected)?,
        };
        decode_sample(&sample)
    }

    /// Non-blocking receive: `Ok(None)` when no sample is queued.
    pub fn try_recv(&self) -> Result<Option<TelemetrySample>> {
        let sub = self.sub.as_ref().ok_or(NetError::NotSubscribed)?;
        match sub.try_recv().map_err(|_| NetError::Disconnected)? {
            Some(s) => Ok(Some(decode_sample(&s)?)),
            None => Ok(None),
        }
    }
}

fn decode_sample(sample: &zenoh::sample::Sample) -> Result<TelemetrySample> {
    let bytes = sample.payload().to_bytes();
    decode_telemetry(&bytes).map_err(|_| NetError::Decode)
}

/// Sim-side command intake — subscribes `astrum/command/**` (fan-in from every
/// controller) and drains decoded commands non-blockingly each tick.
pub struct CommandReceiver {
    sub: FifoSubscriber,
}

impl CommandReceiver {
    /// Declare the command subscriber on `session`.
    pub fn bind(session: &Session) -> Result<Self> {
        let sub =
            session.z.declare_subscriber(format!("{COMMAND_PREFIX}/**")).wait().map_err(zerr)?;
        Ok(Self { sub })
    }

    /// Drain every queued command without blocking. Undecodable payloads are
    /// skipped so one malformed frame can't wedge the intake.
    pub fn drain(&self) -> Vec<CommandMsg> {
        let mut out = Vec::new();
        while let Ok(Some(sample)) = self.sub.try_recv() {
            let bytes = sample.payload().to_bytes();
            if let Ok(cmd) = decode_command(&bytes) {
                out.push(cmd);
            }
        }
        out
    }
}

/// Controller-side command sender. Publishes on `astrum/command/<target-id>`
/// with `Block` congestion control so a control input is never silently dropped
/// (reliable over the TCP transport, matching ZMQ `PUSH`/`PULL`).
pub struct CommandSender {
    session: Session,
}

impl CommandSender {
    /// Build a sender on `session`.
    pub fn connect(session: &Session) -> Self {
        Self { session: session.clone() }
    }

    /// Send one command to its target aircraft.
    pub fn send(&self, cmd: &CommandMsg) -> Result<()> {
        let payload = encode_command(cmd);
        self.session
            .z
            .put(command_key(cmd.target_id), payload)
            .congestion_control(CongestionControl::Block)
            .wait()
            .map_err(zerr)
    }
}

// --- Drone racing: course + progress ----------------------------------------
//
// The racing environment (`astrum-race`) publishes the drone's own state as plain
// [`TelemetrySample`]s on the telemetry key and takes [`CommandMsg`]s on the
// command key, exactly like the fleet sim. Only the *task* state needs its own
// channel, and it gets its own key space (`astrum/race/**`) so nothing above
// changes.

/// A racing subscriber's channel: a **ring of one**, not a FIFO queue.
///
/// Both racing channels carry **state**, not events: the course is a static
/// snapshot republished on a timer, and progress figures are cumulative counters.
/// So a backlog is pure staleness — the newest message subsumes every one behind
/// it, and a reader that falls behind should skip rather than crawl through
/// history. A ring of one makes that a property of the transport: the handler keeps
/// the latest sample and drops what it superseded, so `try_recv` is inherently
/// latest-wins and needs no drain loop.
///
/// It also removes a foot-gun. Zenoh's default FIFO handler **blocks the
/// publisher** when its channel fills and nobody reads — `CongestionControl::Drop`
/// governs the network queues, not a local handler — so a same-process subscriber
/// that stops draining would stall the 120 Hz physics loop after a couple of
/// hundred ticks. A ring cannot do that to anyone.
type RingSubscriber =
    zenoh::pubsub::Subscriber<zenoh::handlers::RingChannelHandler<zenoh::sample::Sample>>;

/// Environment-side publisher for the racing task state: the course on
/// `astrum/race/course/<drone-id>` and progress on `astrum/race/state/<drone-id>`.
///
/// Both go out best-effort (`Drop`), like telemetry: this runs inside the physics
/// loop, and blocking a 120 Hz step on a slow subscriber would be worse than
/// losing a frame. That is safe for progress because it is counters rather than
/// events, and safe for the course because the environment **republishes** it
/// periodically — a lost or late-joined course arrives on the next cycle.
pub struct RacePublisher {
    session: Session,
}

impl RacePublisher {
    /// Build a publisher on `session`.
    pub fn new(session: &Session) -> Self {
        Self { session: session.clone() }
    }

    /// Publish the course on `astrum/race/course/<course.drone_id>`.
    pub fn publish_course(&self, course: &RaceCourseMsg) -> Result<()> {
        self.put(race_course_key(course.drone_id), encode_race_course(course))
    }

    /// Publish one progress sample on `astrum/race/state/<state.drone_id>`.
    pub fn publish_state(&self, state: &RaceStateMsg) -> Result<()> {
        self.put(race_state_key(state.drone_id), encode_race_state(state))
    }

    fn put(&self, key: String, payload: Vec<u8>) -> Result<()> {
        self.session
            .z
            .put(key, payload)
            .congestion_control(CongestionControl::Drop)
            .wait()
            .map_err(zerr)
    }
}

/// Controller-side reader for the racing task state — the counterpart to
/// [`RacePublisher`]. One type holds both channels, because a racing controller
/// wants both: the course tells it where the gates are, the progress tells it
/// which one it is flying at.
///
/// Construction mirrors [`TelemetrySubscriber`]: [`connect`](Self::connect) opens
/// the reader, then [`subscribe`](Self::subscribe) / [`subscribe_all`](Self::subscribe_all)
/// declares the two Zenoh subscribers.
pub struct RaceSubscriber {
    session: Session,
    course: Option<RingSubscriber>,
    state: Option<RingSubscriber>,
}

impl RaceSubscriber {
    /// Prepare a reader on `session`. Nothing is delivered until you subscribe.
    pub fn connect(session: &Session) -> Result<Self> {
        Ok(Self { session: session.clone(), course: None, state: None })
    }

    /// Subscribe to one racing drone's course and progress.
    pub fn subscribe(&mut self, drone_id: u32) -> Result<()> {
        self.declare(race_course_key(drone_id), race_state_key(drone_id))
    }

    /// Subscribe to every racing drone's course and progress.
    pub fn subscribe_all(&mut self) -> Result<()> {
        self.declare(format!("{RACE_COURSE_PREFIX}/*"), format!("{RACE_STATE_PREFIX}/*"))
    }

    fn declare(&mut self, course_key: String, state_key: String) -> Result<()> {
        self.course = Some(self.declare_ring(course_key)?);
        self.state = Some(self.declare_ring(state_key)?);
        Ok(())
    }

    /// Declare one [`RingSubscriber`]. Capacity **1**: these are state channels, so
    /// the only sample worth holding is the newest.
    fn declare_ring(&self, key: String) -> Result<RingSubscriber> {
        self.session
            .z
            .declare_subscriber(key)
            .with(zenoh::handlers::RingChannel::new(1))
            .wait()
            .map_err(zerr)
    }

    /// Non-blocking read of the latest course, or `Ok(None)` if none has arrived
    /// since the last call. Because the environment republishes the course on a
    /// timer, a controller can simply poll this until it returns `Some`.
    pub fn try_recv_course(&self) -> Result<Option<RaceCourseMsg>> {
        let sub = self.course.as_ref().ok_or(NetError::NotSubscribed)?;
        match sub.try_recv().map_err(|_| NetError::Disconnected)? {
            Some(s) => {
                Ok(Some(decode_race_course(&s.payload().to_bytes()).map_err(|_| NetError::Decode)?))
            }
            None => Ok(None),
        }
    }

    /// Non-blocking read of the latest progress sample.
    pub fn try_recv_state(&self) -> Result<Option<RaceStateMsg>> {
        let sub = self.state.as_ref().ok_or(NetError::NotSubscribed)?;
        match sub.try_recv().map_err(|_| NetError::Disconnected)? {
            Some(s) => {
                Ok(Some(decode_race_state(&s.payload().to_bytes()).map_err(|_| NetError::Decode)?))
            }
            None => Ok(None),
        }
    }
}
