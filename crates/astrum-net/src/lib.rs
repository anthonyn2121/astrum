//! `astrum-net` — the ASTRUM Zenoh transport layer.
//!
//! One Zenoh session per process carries every stream, each keyed by aircraft id
//! (see [`keys`]):
//!
//! * **Telemetry** — the sim publishes each [`Telemetry`] on
//!   `astrum/telemetry/<id>` through a per-aircraft declared publisher.
//!   [`LatestTelemetry`] keeps the newest sample per aircraft for controllers and
//!   viewers: a control law must act on the freshest state, never on a queue of
//!   stale frames. [`SampledTelemetry`] keeps a bounded, thinned trail of every
//!   aircraft's samples instead, for viewers that draw history (the web bridge
//!   at faster than real time).
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
//! * **Race** — a race run publishes each aircraft's [`RaceCourse`] on
//!   `astrum/race/course/<id>` and its [`RaceState`] on `astrum/race/state/<id>`
//!   ([`RacePublisher`], latest-value QoS); [`RaceSubscriber`] hands out the newest
//!   of each. Its gate events travel on `astrum/events`.
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
//! router offers Zenoh's low-latency transport and every client asks for it for
//! its publish/subscribe link — the C++ and Python controllers (every lockstep
//! round trip counts) and [`Session::connect`] (recorder, web bridge, ctl),
//! which on the default transport could lose all telemetry to a Zenoh 1.10
//! congestion bug (see `LOW_LATENCY`). It has ONE lane per link, no batching and
//! no fragmentation, so priorities reorder nothing — messages arrive in the
//! order the sim sent them — nothing is dropped (a client that falls behind
//! slows the sender, so `CongestionControl::Drop` drops nothing here), and a
//! message must fit in one transport frame, which the stream's samples and
//! events do by a wide margin. Queries and their replies (scenario, status,
//! control) go over a second, default-transport link, where a large reply is
//! fragmented and — being `Block`, not droppable — cannot trip the latch.
//!
//! Every call uses Zenoh's synchronous `.wait()` resolver; Zenoh runs its own
//! runtime internally, so these wrappers drop into synchronous code as-is.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use astrum_proto::{
    decode_command, decode_event, decode_race_course, decode_race_state, decode_telemetry,
    decode_tick_end, encode_command, encode_race_course, encode_race_state, encode_telemetry,
    encode_tick_end,
};
pub use astrum_proto::{
    Command, Control, Event, EventKind, RaceCourse, RaceState, Telemetry, TickEnd,
};
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
    /// `astrum/race/course/<id>`: the race course aircraft `<id>` flies.
    pub const RACE_COURSE_PREFIX: &str = "astrum/race/course";
    /// `astrum/race/state/<id>`: aircraft `<id>`'s race progress.
    pub const RACE_STATE_PREFIX: &str = "astrum/race/state";

    pub fn telemetry(id: u32) -> String {
        format!("{TELEMETRY_PREFIX}/{id}")
    }

    pub fn command(id: u32) -> String {
        format!("{COMMAND_PREFIX}/{id}")
    }

    pub fn alive(id: u32) -> String {
        format!("{ALIVE_PREFIX}/{id}")
    }

    pub fn race_course(id: u32) -> String {
        format!("{RACE_COURSE_PREFIX}/{id}")
    }

    pub fn race_state(id: u32) -> String {
        format!("{RACE_STATE_PREFIX}/{id}")
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

/// A Zenoh session. Cheap to clone (an `Arc` or two inside).
///
/// A client ([`connect`](Self::connect)) holds two links to the hub: `z`, on
/// the low-latency transport, carries every subscription, publication and
/// liveliness token; `q`, on the default transport, carries queries and their
/// replies. A hub ([`listen`](Self::listen)) and an in-process session need only
/// `z`.
#[derive(Clone)]
pub struct Session {
    z: zenoh::Session,
    q: Option<zenoh::Session>,
}

/// Zenoh's low-latency transport — no batching, no QoS lanes, no fragmentation:
/// the sim's router offers it and every client asks for it for its streams (the
/// C++ and Python SDKs set the same two keys). Besides the shorter lockstep
/// round trip it avoids a Zenoh 1.10 bug in the default transport: on a link
/// without QoS lanes (the router turns them off for this transport) the
/// transmission pipeline marks congestion on the `Priority::DEFAULT` bit but
/// clears the bit of queue 0 (`zenoh-transport` `src/common/pipeline.rs`,
/// `push_network_message` / `refill`), so once a client fell behind and one
/// droppable sample was dropped, every later droppable message to it —
/// telemetry, commands — was dropped for the rest of the session. (1.10.2, the
/// latest, has the same code.) On this transport a write to a client that falls
/// behind blocks the sender instead (nothing is dropped), and **a client that
/// stops reading — a suspended recorder, web bridge, ctl or controller —
/// freezes the sim, and with it the control plane, until it reads again or its
/// connection dies**; the sim's command lease never releases it. A message must
/// also fit in one frame (the TCP link's Zenoh MTU, ~48 KiB on loopback, never
/// below 32 KiB), which is why queries do not use it: only droppable messages
/// can latch, and query replies are not droppable, so they travel the
/// fragmenting default transport.
const LOW_LATENCY: [(&str, &str); 2] =
    [("transport/unicast/lowlatency", "true"), ("transport/unicast/qos/enabled", "false")];

impl Session {
    /// Open a session that LISTENS on `endpoint` (the sim / bag-replay role) as a
    /// Zenoh **router**: every other process connects only to this hub, and peers
    /// do not forward for each other, so the hub must route — otherwise one
    /// client's publications (a controller's commands) would never reach another
    /// client (the recorder). Multicast scouting is off: Docker bridge networks
    /// don't pass multicast, so clients connect by explicit endpoint. It offers
    /// the low-latency transport (`LOW_LATENCY`) to every client that asks.
    pub fn listen(endpoint: &str) -> Result<Self> {
        Self::open(
            &[
                ("scouting/multicast/enabled", "false".into()),
                ("mode", r#""router""#.into()),
                ("listen/endpoints", format!(r#"["{endpoint}"]"#)),
            ],
            &LOW_LATENCY,
        )
    }

    /// Open a session that CONNECTS to the hub at `endpoint` (recorder / web /
    /// ctl) as a Zenoh **client**: the hub routes everything for it, including
    /// other clients' publications. Its streams ride the low-latency transport
    /// (`LOW_LATENCY`); a second link on the default transport carries its
    /// queries, so a reply of any size gets through. Waits up to `timeout` for
    /// the hub (a sim still loading its aircraft), then fails; once connected,
    /// Zenoh reconnects by itself if the hub restarts.
    pub fn connect(endpoint: &str, timeout: Duration) -> Result<Self> {
        let timeout_ms = i64::try_from(timeout.as_millis()).unwrap_or(i64::MAX);
        let patches = [
            ("scouting/multicast/enabled", "false".into()),
            ("mode", r#""client""#.into()),
            ("connect/endpoints", format!(r#"["{endpoint}"]"#)),
            ("connect/timeout_ms", timeout_ms.to_string()),
        ];
        let z = Self::open_raw(&patches, &LOW_LATENCY)?;
        // The hub is up now, so this one connects at once.
        let q = Self::open_raw(&patches, &[])?;
        Ok(Self { z, q: Some(q) })
    }

    /// An isolated in-process session (no endpoints) whose publishers deliver to
    /// its own subscribers — for tests.
    pub fn in_process() -> Result<Self> {
        Self::open(
            &[("scouting/multicast/enabled", "false".into()), ("mode", r#""peer""#.into())],
            &[],
        )
    }

    fn open(patches: &[(&str, String)], transport: &[(&str, &str)]) -> Result<Self> {
        Ok(Self { z: Self::open_raw(patches, transport)?, q: None })
    }

    fn open_raw(patches: &[(&str, String)], transport: &[(&str, &str)]) -> Result<zenoh::Session> {
        let mut config = zenoh::Config::default();
        let transport = transport.iter().map(|&(key, value)| (key, value.to_owned()));
        for (key, value) in patches.iter().cloned().chain(transport) {
            config.insert_json5(key, &value).map_err(zerr)?;
        }
        zenoh::open(config).wait().map_err(zerr)
    }

    /// The session queries go through: the client's default-transport link,
    /// else the one session.
    fn queries(&self) -> &zenoh::Session {
        self.q.as_ref().unwrap_or(&self.z)
    }

    /// Close the session (both links of a client). Other clones of it stop
    /// working too.
    pub fn close(&self) -> Result<()> {
        let q = self.q.as_ref().map(|q| q.close().wait().map_err(zerr));
        self.z.close().wait().map_err(zerr)?;
        q.unwrap_or(Ok(()))
    }

    /// Whether every link of the session reaches a hub right now (a client's
    /// stream link and its query link; a hub or in-process session has one).
    pub fn linked(&self) -> bool {
        let up = |z: &zenoh::Session| z.info().routers_zid().wait().next().is_some();
        up(&self.z) && self.q.as_ref().is_none_or(up)
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

/// Share of a [`SampledTelemetry`] period two samples may fall short of it and
/// still count as spaced: sim times are `tick / rate_hz`, so a sample exactly
/// one period later can round a hair below it.
const SPACING_SLACK: f64 = 1e-6;

/// One aircraft's trail in a [`SampledTelemetry`].
#[derive(Default)]
struct Trail {
    /// Samples not yet taken, oldest first. Each is at least one trail period
    /// of sim time after the one before it (the first: after the last one
    /// taken), except the last, which is always the newest sample received.
    pending: VecDeque<Telemetry>,
    /// The newest sample received.
    newest: Option<Telemetry>,
    /// Sim time of the last sample taken (`None` until one is, and after the
    /// aircraft's clock went backwards).
    taken: Option<f64>,
    /// Times the cap halved `pending` since the last take: the trail's period
    /// is the base period doubled this many times.
    halvings: u32,
}

impl Trail {
    /// Add `t`, thinning to one sample per trail period of sim time (`period`
    /// doubled once per halving). When more than `cap` would be pending, keep
    /// every other one — the newest always — and double the period, so the
    /// batch still spans the whole stretch since the last take, at a coarser
    /// resolution. Returns the samples the halving dropped.
    fn add(&mut self, t: Telemetry, period: f64, cap: usize) -> u64 {
        if self.newest.is_some_and(|n| t.sim_time_s < n.sim_time_s) {
            // The clock went backwards: a restarted sim or replay. The old
            // run's samples go, so a trail is always time-ordered.
            self.pending.clear();
            self.taken = None;
            self.halvings = 0;
        }
        self.newest = Some(t);
        let len = self.pending.len();
        if len == 0 {
            self.pending.push_back(t);
            return 0;
        }
        let period = period * 2f64.powi(self.halvings.try_into().unwrap_or(i32::MAX));
        let before = if len >= 2 { Some(self.pending[len - 2].sim_time_s) } else { self.taken };
        let tail = &mut self.pending[len - 1];
        let spaced = before.is_none_or(|b| tail.sim_time_s - b >= period * (1.0 - SPACING_SLACK));
        if !spaced || tail.sim_time_s == t.sim_time_s {
            *tail = t; // the tail only stood in for the newest sample
            return 0;
        }
        self.pending.push_back(t);
        if self.pending.len() > cap {
            return self.halve(cap);
        }
        0
    }

    /// Keep every other pending sample, counting back from the newest (which
    /// stays), and double the period; returns how many went. Kept samples are
    /// two old periods apart, so they still satisfy the doubled one. The oldest
    /// sample stays too (when there is room under the cap), so a batch starts
    /// where the stretch since the last take starts.
    fn halve(&mut self, cap: usize) -> u64 {
        let before = self.pending.len();
        let mut kept: Vec<Telemetry> = self.pending.iter().rev().step_by(2).copied().collect();
        kept.reverse();
        if kept.len() < cap && kept.first() != self.pending.front() {
            kept.insert(0, self.pending[0]);
        }
        self.pending = kept.into();
        self.halvings = self.halvings.saturating_add(1);
        (before - self.pending.len()) as u64
    }
}

#[derive(Default)]
struct SampledState {
    trails: BTreeMap<u32, Trail>,
    generation: u64,
    decode_errors: u64,
    capped: u64,
}

/// A thinned trail of every aircraft's samples, taken in batches: a viewer that
/// draws history at faster than real time gets every aircraft's flight at an
/// even sim-time resolution, not just whatever sample was newest when it
/// looked. Samples are decoded and thinned on Zenoh's receive thread, so a slow
/// consumer never blocks the transport, and memory stays bounded:
///
/// * **thinned by sim time** — an aircraft's pending samples are at least one
///   trail period of sim time apart (`period`, doubled once per halving, below),
///   except that the newest sample received is always pending too (it replaces
///   a previous newest that was closer than a period to the sample before it).
///   So between two takes an aircraft gains at most one sample per period plus
///   its newest, and consecutive kept samples are never more than a trail period
///   plus one publication interval apart.
/// * **capped by halving** — at most `cap` samples per aircraft wait to be
///   taken. A sim far faster than real time would pass the cap within one
///   batch; rather than drop the oldest (a batch that covers only its newest
///   seconds), the trail keeps every other pending sample (the newest always,
///   the oldest when there is room) and doubles its period, again each time
///   the cap is passed, so a batch spans the whole stretch since the last take
///   with between about `cap / 2` and `cap` samples, at a power-of-two
///   multiple of the sample period. A take resets the period.
///   ([`capped`](Self::capped) counts the samples halving dropped.)
/// * **time-ordered** — a sample older than the aircraft's newest (a restarted
///   sim or replay) discards that aircraft's pending samples first.
///
/// It subscribes like [`LatestTelemetry`]; the callback is O(1) per sample, so
/// on the low-latency transport — where a reader that falls behind slows the
/// sim rather than losing samples — it keeps up with a sim far faster than
/// real time.
pub struct SampledTelemetry {
    shared: Arc<(Mutex<SampledState>, Condvar)>,
    seen_generation: u64,
    _sub: Subscriber<()>,
}

impl SampledTelemetry {
    /// Subscribe to `selection`, keeping samples at least `period_s` of sim
    /// time apart and at most `cap` (≥ 1) pending per aircraft.
    pub fn subscribe(
        session: &Session,
        selection: Selection,
        period_s: f64,
        cap: usize,
    ) -> Result<Self> {
        let key = match selection {
            Selection::Aircraft(id) => keys::telemetry(id),
            Selection::Fleet => keys::TELEMETRY_ALL.to_owned(),
        };
        let cap = cap.max(1);
        let shared: Arc<(Mutex<SampledState>, Condvar)> = Arc::default();
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
                        let capped =
                            st.trails.entry(t.aircraft_id).or_default().add(t, period_s, cap);
                        st.capped += capped;
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

    fn state(&self) -> std::sync::MutexGuard<'_, SampledState> {
        self.shared.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Payloads that failed to decode so far.
    pub fn decode_errors(&self) -> u64 {
        self.state().decode_errors
    }

    /// Samples the cap dropped so far.
    pub fn capped(&self) -> u64 {
        self.state().capped
    }

    /// Block until at least one sample arrives after the last `wait` or
    /// [`take`](Self::take).
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

    /// The batch since the last take, ordered by aircraft id and, per aircraft,
    /// by sim time: every aircraft's pending samples, or — for one with none —
    /// its newest sample again, so a batch always covers every aircraft seen.
    /// Empty until a sample arrived.
    pub fn take(&mut self) -> Vec<Telemetry> {
        let mut st = self.state();
        let generation = st.generation;
        let mut batch = Vec::new();
        for trail in st.trails.values_mut() {
            match trail.pending.back() {
                Some(last) => {
                    trail.taken = Some(last.sim_time_s);
                    trail.halvings = 0;
                    batch.extend(trail.pending.drain(..));
                }
                None => batch.extend(trail.newest),
            }
        }
        drop(st);
        self.seen_generation = generation;
        batch
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

// --- Race -----------------------------------------------------------------------

/// Sim-side publisher of one aircraft's race messages: [`RaceCourse`] on
/// `astrum/race/course/<id>` and [`RaceState`] on `astrum/race/state/<id>`. Both are
/// latest-value state (the ASTRUM [`qos`]: drop, express, `InteractiveHigh`).
pub struct RacePublisher {
    course: Publisher<'static>,
    state: Publisher<'static>,
}

impl RacePublisher {
    pub fn new(session: &Session, aircraft_id: u32) -> Result<Self> {
        Ok(Self {
            course: session.declare_publisher(keys::race_course(aircraft_id))?,
            state: session.declare_publisher(keys::race_state(aircraft_id))?,
        })
    }

    pub fn publish_course(&self, c: &RaceCourse) -> Result<()> {
        self.course.put(encode_race_course(c)).wait().map_err(zerr)
    }

    pub fn publish_state(&self, s: &RaceState) -> Result<()> {
        self.state.put(encode_race_state(s)).wait().map_err(zerr)
    }
}

/// The newest decoded message of one race stream, until taken.
struct Latest<T> {
    value: Mutex<LatestValue<T>>,
}

struct LatestValue<T> {
    value: Option<T>,
    decode_errors: u64,
}

impl<T: Send + 'static> Latest<T> {
    /// Subscribe to `key`, decoding each payload on Zenoh's receive thread.
    fn subscribe(
        session: &Session,
        key: String,
        decode: fn(&[u8]) -> Result<T, astrum_proto::DecodeError>,
    ) -> Result<(Arc<Self>, Subscriber<()>)> {
        let shared =
            Arc::new(Self { value: Mutex::new(LatestValue { value: None, decode_errors: 0 }) });
        let sink = Arc::clone(&shared);
        let sub = session
            .z
            .declare_subscriber(key)
            .callback(move |sample: Sample| {
                let decoded = decode(&sample.payload().to_bytes());
                let mut slot = sink.state();
                match decoded {
                    Ok(v) => slot.value = Some(v),
                    Err(_) => slot.decode_errors += 1,
                }
            })
            .wait()
            .map_err(zerr)?;
        Ok((shared, sub))
    }

    fn state(&self) -> std::sync::MutexGuard<'_, LatestValue<T>> {
        self.value.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn take(&self) -> Option<T> {
        self.state().value.take()
    }
}

/// Subscriber of one aircraft's race messages. Each `try_recv_*` returns the newest
/// message that arrived since the previous call (latest wins), else `None`.
pub struct RaceSubscriber {
    course: Arc<Latest<RaceCourse>>,
    state: Arc<Latest<RaceState>>,
    _course_sub: Subscriber<()>,
    _state_sub: Subscriber<()>,
}

impl RaceSubscriber {
    pub fn new(session: &Session, aircraft_id: u32) -> Result<Self> {
        let (course, _course_sub) =
            Latest::subscribe(session, keys::race_course(aircraft_id), decode_race_course)?;
        let (state, _state_sub) =
            Latest::subscribe(session, keys::race_state(aircraft_id), decode_race_state)?;
        Ok(Self { course, state, _course_sub, _state_sub })
    }

    pub fn try_recv_course(&self) -> Option<RaceCourse> {
        self.course.take()
    }

    pub fn try_recv_state(&self) -> Option<RaceState> {
        self.state.take()
    }

    /// Race payloads (course and state) that failed to decode so far.
    pub fn decode_errors(&self) -> u64 {
        self.course.state().decode_errors + self.state.state().decode_errors
    }
}

// --- Scenario -----------------------------------------------------------------

/// Answers `get astrum/scenario` with the sim's scenario TOML for as long as it
/// is alive.
pub struct ScenarioServer {
    _queryable: Queryable<()>,
}

impl ScenarioServer {
    /// Serve `scenario_toml`. A client receives it over its query link, so a
    /// scenario of any size arrives (the default transport fragments).
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
        let started = Instant::now();
        // A query waits out the whole of `left` for its reply — a large
        // scenario on a slow link takes longer than one retry period — but one
        // nobody answers finishes at once, and the retry sleep below paces it.
        let replies = session.queries().get(keys::SCENARIO).timeout(left).wait().map_err(zerr)?;
        while let Ok(reply) = replies.recv() {
            if let Ok(sample) = reply.result() {
                let text = String::from_utf8(sample.payload().to_bytes().into_owned())
                    .map_err(|e| NetError::BadScenario(e.to_string()))?;
                return Ok(Some(text));
            }
        }
        // No queryable yet: the query finished early; wait out the attempt.
        std::thread::sleep(SCENARIO_RETRY.saturating_sub(started.elapsed()).min(left));
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

/// The one rule for a sim speed: `0` (unpaced), or finite and at least
/// `min_speed`.
pub fn speed_ok(x: f64, min_speed: f64) -> bool {
    x == 0.0 || (x.is_finite() && x >= min_speed)
}

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
            ControlRequest::Speed(x) if !speed_ok(x, min_speed) => {
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
    let replies = session.queries().get(selector).timeout(timeout).wait().map_err(zerr)?;
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
