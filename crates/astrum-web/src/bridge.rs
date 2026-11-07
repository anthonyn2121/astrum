//! The [`Bridge`]: the browser-facing feeds of one `astrum-web` server, fed by
//! whichever sim (or replay) it is attached to.
//!
//! The bridge owns its channels for its whole lifetime; attachments come and
//! go. [`Bridge::attach`] points it at a Zenoh endpoint: a background thread
//! connects (retrying until a sim answers, so an attachment can exist while
//! nothing is connected) and runs three pumps that publish into the channels:
//!
//! * **snapshots** (latest-value, [`watch`]): the newest sample of every
//!   aircraft as ONE `{"type":"frames",…}` message, at most one per
//!   [`BridgeConfig::min_period`] and only when something changed. A browser
//!   that falls behind skips to the newest one; one that connects mid-run gets
//!   the current snapshot at once. The payload is a [`Utf8Bytes`], so handing it
//!   to N browsers is N reference-count bumps, not N copies.
//! * **events** (in order, [`broadcast`]): each batch of `astrum/events` as one
//!   `{"type":"events",…}` message, tagged with its run. A browser gets only the
//!   batches sent after it subscribed; one that falls more than
//!   [`EVENT_BACKLOG`] batches behind loses the oldest of them.
//! * **models** (latest-value, [`watch`]): the scenario's `[[aircraft]]` models
//!   as one `{"type":"models",…}` message, polled from `astrum/scenario` every
//!   [`BridgeConfig::models_poll`] and replaced only when it differs. A poll
//!   nobody answers keeps the last message.
//!
//! Every attach or detach starts a new **run**: the previous attachment's
//! pumps stop and its session closes, the snapshot and models are cleared (a
//! new attachment starts from a fresh [`LatestTelemetry`], so no aircraft of
//! the previous sim survives), and a `{"type":"run",…}` note is published.
//! Pumps publish only while their run is current, checked under the same lock
//! the switch takes, so nothing from an old run lands after its note.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use astrum_net::zenoh::Wait;
use astrum_net::{fetch_scenario, EventReceiver, LatestTelemetry, NetError, Selection, Session};
use astrum_proto::Telemetry;
use axum::extract::ws::Utf8Bytes;
use serde::Serialize;
use tokio::sync::{broadcast, watch};

use crate::{aircraft_models, events_message, models_message, run_message, snapshot_message};

/// The newest encoded fleet snapshot (`None` until the current run has one).
pub type Snapshot = Option<Utf8Bytes>;

/// Event batches buffered per browser before the slowest one starts losing them.
pub const EVENT_BACKLOG: usize = 1024;
/// How long one connection attempt waits for the sim before trying again.
const CONNECT_ATTEMPT: Duration = Duration::from_secs(1);
/// Shortest gap between the starts of two connection attempts (an endpoint
/// that fails at once is not hammered).
const CONNECT_RETRY: Duration = Duration::from_millis(500);
/// How often an idle pump re-checks whether its run is still current; bounds
/// how long a superseded attachment lingers.
const IDLE_POLL: Duration = Duration::from_millis(200);
/// How often a connected attachment re-checks that the sim is still there.
const LINK_POLL: Duration = Duration::from_millis(250);
/// How long one scenario poll waits for the sim's reply: short, because no
/// reply simply keeps the models already sent.
const SCENARIO_TIMEOUT: Duration = Duration::from_secs(1);

/// Pacing of the bridge's pumps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BridgeConfig {
    /// Shortest gap between two fleet snapshots.
    pub min_period: Duration,
    /// How often the scenario's aircraft models are re-fetched.
    pub models_poll: Duration,
}

/// What the bridge is attached to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BridgeInfo {
    /// The attached endpoint; `None` while detached.
    pub endpoint: Option<String>,
    /// The current run id: 0 before the first attach, bumped by every attach
    /// and detach.
    pub run: u64,
    /// Whether a sim (or replay) is reachable on `endpoint` right now.
    pub connected: bool,
}

/// The current run, as browsers hear of it.
#[derive(Debug, Clone)]
pub struct RunNote {
    pub run: u64,
    /// `{"type":"run","run":N,"endpoint":…}` ([`run_message`]).
    pub message: Utf8Bytes,
}

/// One `{"type":"events",…}` message and the run it came from.
#[derive(Debug, Clone)]
pub struct EventBatch {
    pub run: u64,
    pub message: Utf8Bytes,
}

/// The server's browser-facing feeds and its current attachment. Cheap to
/// clone; the last clone dropped detaches.
#[derive(Clone)]
pub struct Bridge {
    shared: Arc<Shared>,
    _owner: Arc<Owner>,
}

/// Detaches when the last [`Bridge`] handle goes (attachment threads hold only
/// [`Shared`], so they cannot keep the bridge alive).
struct Owner(Arc<Shared>);

impl Drop for Owner {
    fn drop(&mut self) {
        self.0.switch(None);
    }
}

struct Shared {
    config: BridgeConfig,
    snapshots: watch::Sender<Snapshot>,
    events: broadcast::Sender<EventBatch>,
    models: watch::Sender<Option<Utf8Bytes>>,
    runs: watch::Sender<RunNote>,
    current: Mutex<Current>,
}

struct Current {
    run: u64,
    attachment: Option<Attachment>,
}

struct Attachment {
    endpoint: String,
    /// The connected session (set once a sim answered).
    session: Option<Session>,
    /// A sim is reachable through `session` right now.
    connected: bool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Current> {
        self.current.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Start a new run attached to `endpoint` (or detached): clear the feeds
    /// and announce it. Returns the new run id.
    fn switch(&self, endpoint: Option<&str>) -> u64 {
        let mut current = self.lock();
        current.run += 1;
        let run = current.run;
        // The old attachment's thread still holds its own session handle (it
        // closes it once it sees the switch), so dropping ours here never closes
        // a Zenoh session on the caller's (possibly async) thread.
        let old = std::mem::replace(
            &mut current.attachment,
            endpoint.map(|e| Attachment {
                endpoint: e.to_owned(),
                session: None,
                connected: false,
            }),
        );
        self.snapshots.send_replace(None);
        self.models.send_replace(None);
        self.runs
            .send_replace(RunNote { run, message: Utf8Bytes::from(run_message(run, endpoint)) });
        drop(current);
        drop(old);
        run
    }
}

impl Bridge {
    /// A detached bridge (run 0).
    pub fn new(config: BridgeConfig) -> Self {
        let (snapshots, _) = watch::channel(None);
        let (events, _) = broadcast::channel(EVENT_BACKLOG);
        let (models, _) = watch::channel(None);
        let (runs, _) =
            watch::channel(RunNote { run: 0, message: Utf8Bytes::from(run_message(0, None)) });
        let shared = Arc::new(Shared {
            config,
            snapshots,
            events,
            models,
            runs,
            current: Mutex::new(Current { run: 0, attachment: None }),
        });
        Self { _owner: Arc::new(Owner(Arc::clone(&shared))), shared }
    }

    /// Attach to the sim (or replay) on `endpoint`, replacing any previous
    /// attachment, and return the new run id. Connecting happens in the
    /// background, retrying until a sim answers; never blocks.
    pub fn attach(&self, endpoint: &str) -> u64 {
        let run = self.shared.switch(Some(endpoint));
        let feed = Feed { shared: Arc::clone(&self.shared), run };
        let endpoint = endpoint.to_owned();
        std::thread::Builder::new()
            .name(format!("bridge-run-{run}"))
            .spawn(move || feed.serve(&endpoint))
            .expect("spawning the bridge thread");
        run
    }

    /// Drop the current attachment (if any) and start an empty run.
    pub fn detach(&self) {
        self.shared.switch(None);
    }

    /// The current attachment.
    pub fn info(&self) -> BridgeInfo {
        self.link().0
    }

    /// The attached session while a sim is reachable through it, for relaying
    /// requests to the sim. Its Zenoh calls block: make them (and drop the
    /// handle) off the async runtime's worker threads.
    pub fn session(&self) -> Option<Session> {
        self.link().1
    }

    /// [`info`](Self::info) and [`session`](Self::session) read together, so
    /// they describe the same attachment.
    pub fn link(&self) -> (BridgeInfo, Option<Session>) {
        let current = self.shared.lock();
        let attachment = current.attachment.as_ref();
        let connected = attachment.is_some_and(|a| a.connected);
        let info = BridgeInfo {
            endpoint: attachment.map(|a| a.endpoint.clone()),
            run: current.run,
            connected,
        };
        let session = attachment.filter(|_| connected).and_then(|a| a.session.clone());
        (info, session)
    }

    /// The newest fleet snapshot of the current run.
    pub fn snapshots(&self) -> watch::Receiver<Snapshot> {
        self.shared.snapshots.subscribe()
    }

    /// Event batches from now on.
    pub fn events(&self) -> broadcast::Receiver<EventBatch> {
        self.shared.events.subscribe()
    }

    /// The current run's aircraft-models message, once known.
    pub fn models(&self) -> watch::Receiver<Option<Utf8Bytes>> {
        self.shared.models.subscribe()
    }

    /// The current run.
    pub fn runs(&self) -> watch::Receiver<RunNote> {
        self.shared.runs.subscribe()
    }
}

/// One attachment's view of the bridge: it may publish only while its run is
/// current.
struct Feed {
    shared: Arc<Shared>,
    run: u64,
}

/// When one connection's pumps stop: the run was superseded, or a pump failed.
struct Stop<'a> {
    feed: &'a Feed,
    halt: &'a AtomicBool,
}

impl Stop<'_> {
    fn stopped(&self) -> bool {
        self.halt.load(Ordering::Relaxed) || !self.feed.current()
    }
}

impl Feed {
    fn current(&self) -> bool {
        self.shared.lock().run == self.run
    }

    /// Run `f` on the attachment if this run is still current.
    fn with_attachment(&self, f: impl FnOnce(&mut Attachment)) {
        let mut current = self.shared.lock();
        if current.run == self.run {
            if let Some(attachment) = current.attachment.as_mut() {
                f(attachment);
            }
        }
    }

    /// Publish into the channels if this run is still current; `false` once it
    /// is not.
    fn publish(&self, f: impl FnOnce(&Shared)) -> bool {
        let current = self.shared.lock();
        let live = current.run == self.run;
        if live {
            f(&self.shared);
        }
        live
    }

    /// The attachment thread: connect, pump, reconnect after a transport
    /// failure, until the run is superseded.
    fn serve(&self, endpoint: &str) {
        while let Some(session) = self.connect(endpoint) {
            if let Err(e) = self.pump(&session) {
                eprintln!("[astrum-web] telemetry from {endpoint}: {e}; reconnecting");
            }
            self.with_attachment(|a| {
                a.session = None;
                a.connected = false;
            });
            let _ = session.raw().close().wait();
        }
    }

    /// Connect to `endpoint`, retrying until a sim answers; `None` once the run
    /// is superseded.
    fn connect(&self, endpoint: &str) -> Option<Session> {
        let mut reported = false;
        while self.current() {
            let started = Instant::now();
            match Session::connect(endpoint, CONNECT_ATTEMPT) {
                Ok(session) if self.current() => return Some(session),
                Ok(session) => {
                    let _ = session.raw().close().wait();
                    return None;
                }
                Err(e) => {
                    if !reported {
                        eprintln!("[astrum-web] waiting for a sim on {endpoint} ({e})");
                        reported = true;
                    }
                    std::thread::sleep(CONNECT_RETRY.saturating_sub(started.elapsed()));
                }
            }
        }
        None
    }

    /// Run the three pumps on `session` and keep the link state fresh until
    /// the run is superseded (`Ok`) or a pump fails (its error).
    fn pump(&self, session: &Session) -> Result<(), NetError> {
        let mut latest = LatestTelemetry::subscribe(session, Selection::Fleet)?;
        let events = EventReceiver::subscribe(session)?;
        self.with_attachment(|a| {
            a.session = Some(session.clone());
            a.connected = true;
        });
        let halt = AtomicBool::new(false);
        let stop = &Stop { feed: self, halt: &halt };
        let config = self.shared.config;
        let halting = &|result: Result<(), NetError>| {
            if result.is_err() {
                halt.store(true, Ordering::Relaxed);
            }
            result
        };
        std::thread::scope(|scope| {
            let spawn = std::thread::Builder::new;
            let latest = &mut latest;
            let pumps = [
                spawn()
                    .name("bridge-telemetry".into())
                    .spawn_scoped(scope, move || {
                        halting(pump_snapshots(latest, self, stop, config.min_period))
                    })
                    .expect("spawning the telemetry thread"),
                spawn()
                    .name("bridge-events".into())
                    .spawn_scoped(scope, move || halting(pump_events(&events, self, stop)))
                    .expect("spawning the events thread"),
                spawn()
                    .name("bridge-models".into())
                    .spawn_scoped(scope, move || {
                        halting(pump_models(session, self, stop, config.models_poll))
                    })
                    .expect("spawning the models thread"),
            ];
            while !stop.stopped() {
                let linked = session.raw().info().routers_zid().wait().next().is_some();
                self.with_attachment(|a| a.connected = linked);
                std::thread::sleep(LINK_POLL);
            }
            pumps
                .into_iter()
                .map(|pump| pump.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
                .fold(Ok(()), Result::and)
        })
    }
}

/// Publish fleet snapshots from `latest`: at most one per `min_period`, and
/// only when the fleet state differs from the last one sent. Samples that
/// arrive while throttled are folded into the next snapshot.
fn pump_snapshots(
    latest: &mut LatestTelemetry,
    feed: &Feed,
    stop: &Stop<'_>,
    min_period: Duration,
) -> Result<(), NetError> {
    let mut sent: Vec<Telemetry> = Vec::new();
    let mut sent_at: Option<Instant> = None;
    while !stop.stopped() {
        match latest.wait(IDLE_POLL) {
            Ok(()) => {}
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
        let message = Utf8Bytes::from(snapshot_message(&snapshot));
        if !feed.publish(|s| {
            s.snapshots.send_replace(Some(message));
        }) {
            break;
        }
        sent = snapshot;
        sent_at = Some(Instant::now());
    }
    Ok(())
}

/// Forward every event, one message per batch of events that arrived
/// together, in order. Batches sent while no browser is subscribed are dropped.
fn pump_events(events: &EventReceiver, feed: &Feed, stop: &Stop<'_>) -> Result<(), NetError> {
    while !stop.stopped() {
        let first = match events.recv_timeout(IDLE_POLL) {
            Ok(e) => e,
            Err(NetError::Timeout) => continue,
            Err(e) => return Err(e),
        };
        let mut batch = vec![first];
        batch.extend(events.drain());
        let message = Utf8Bytes::from(events_message(&batch));
        if !feed.publish(|s| {
            let _ = s.events.send(EventBatch { run: feed.run, message });
        }) {
            break;
        }
    }
    Ok(())
}

/// Keep the models channel holding the fleet's aircraft-models message: ask
/// for the scenario at once and then every `poll`, and store a new message
/// only when it differs from the current one, so browsers hear of the fleet
/// once per change (a restarted replay may fly another scenario). A poll
/// nobody answers keeps the last message; so does a reply that is not UTF-8.
fn pump_models(
    session: &Session,
    feed: &Feed,
    stop: &Stop<'_>,
    poll: Duration,
) -> Result<(), NetError> {
    while !stop.stopped() {
        let asked = Instant::now();
        match fetch_scenario(session, SCENARIO_TIMEOUT) {
            Ok(Some(scenario)) => {
                let message = models_message(&aircraft_models(&scenario));
                feed.publish(|s| {
                    s.models.send_if_modified(|current| {
                        if current.as_ref().is_some_and(|c| c.as_str() == message) {
                            return false;
                        }
                        *current = Some(Utf8Bytes::from(message));
                        true
                    });
                });
            }
            Ok(None) | Err(NetError::BadScenario(_)) => {}
            Err(e) => return Err(e),
        }
        while !stop.stopped() {
            let left = poll.saturating_sub(asked.elapsed());
            if left.is_zero() {
                break;
            }
            std::thread::sleep(left.min(IDLE_POLL));
        }
    }
    Ok(())
}
