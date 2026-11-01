//! `astrum-bag` — record the ASTRUM telemetry and command streams to an MCAP
//! file, and replay the telemetry as a stand-in for the sim.
//!
//! A bag is an [MCAP](https://mcap.dev) file with one channel per Zenoh key
//! (topic = key, e.g. `astrum/telemetry/1`, `astrum/command/1`). Every message is
//! the exact Zenoh payload (a Cap'n Proto `Telemetry` / `Command`, never
//! re-encoded); the channel's schema is the full text of `astrum.capnp`. Times:
//!
//! * `log_time` — recorder wall clock at receipt, ns since the Unix epoch;
//! * `publish_time` — sim time in ns for telemetry, the `log_time` for commands.
//!
//! The scenario TOML served by the sim at record start is stored as the metadata
//! record `astrum.scenario` (key `toml`).
//!
//! [`BagWriter`] / [`BagReader`] are the file API; [`Recorder`] and [`replay`]
//! are the record and replay loops behind the `astrum-bag-record` /
//! `astrum-bag-play` binaries, written against a [`Session`] so they run on an
//! in-process session in tests.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use astrum_net::zenoh::pubsub::Subscriber;
use astrum_net::zenoh::Wait;
use astrum_net::{keys, NetError, Session, TelemetryPublisher};
use mcap::records::{MessageHeader, Metadata, Record};

/// Metadata record holding the recorded scenario.
pub const SCENARIO_METADATA: &str = "astrum.scenario";
/// Key of the scenario TOML inside [`SCENARIO_METADATA`].
pub const SCENARIO_KEY: &str = "toml";
/// Message (and schema) encoding of every channel.
pub const ENCODING: &str = "capnp";
/// Schema name of `astrum/telemetry/<id>` channels.
pub const TELEMETRY_SCHEMA: &str = "astrum.wire.Telemetry";
/// Schema name of `astrum/command/<id>` channels.
pub const COMMAND_SCHEMA: &str = "astrum.wire.Command";
/// Full text of the wire schema, stored as every channel's schema data.
pub const SCHEMA_TEXT: &str = include_str!("../../astrum-proto/schema/astrum.capnp");

/// Longest single pause [`replay_schedule`] inserts, so a gap in the recording
/// (or a paused sim) doesn't stall replay.
pub const MAX_REPLAY_GAP: Duration = Duration::from_secs(5);

/// Bag errors.
#[derive(Debug)]
pub enum BagError {
    /// Opening, reading or writing the bag file.
    Io(std::io::Error),
    /// A malformed, truncated or unfinished MCAP file, or an MCAP write failure.
    Mcap(mcap::McapError),
    /// A topic that is not `astrum/telemetry/<id>` or `astrum/command/<id>`.
    UnknownTopic(String),
    /// The Zenoh transport failed.
    Net(NetError),
}

impl fmt::Display for BagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BagError::Io(e) => write!(f, "bag I/O error: {e}"),
            BagError::Mcap(e) => write!(f, "MCAP error: {e}"),
            BagError::UnknownTopic(t) => write!(f, "not an ASTRUM telemetry/command topic: {t:?}"),
            BagError::Net(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BagError::Io(e) => Some(e),
            BagError::Mcap(e) => Some(e),
            BagError::Net(e) => Some(e),
            BagError::UnknownTopic(_) => None,
        }
    }
}

impl From<std::io::Error> for BagError {
    fn from(e: std::io::Error) -> Self {
        BagError::Io(e)
    }
}

impl From<mcap::McapError> for BagError {
    fn from(e: mcap::McapError) -> Self {
        BagError::Mcap(e)
    }
}

impl From<NetError> for BagError {
    fn from(e: NetError) -> Self {
        BagError::Net(e)
    }
}

/// Bag result alias.
pub type Result<T, E = BagError> = std::result::Result<T, E>;

/// What a topic carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopicKind {
    /// `astrum/telemetry/<id>`
    Telemetry(u32),
    /// `astrum/command/<id>`
    Command(u32),
}

impl TopicKind {
    /// Classify a Zenoh key / bag topic; `None` if it is neither stream.
    pub fn of(topic: &str) -> Option<Self> {
        if let Some(id) = keys::id_of(topic, keys::TELEMETRY_PREFIX) {
            Some(TopicKind::Telemetry(id))
        } else {
            keys::id_of(topic, keys::COMMAND_PREFIX).map(TopicKind::Command)
        }
    }
}

/// One recorded message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BagMessage {
    pub topic: String,
    pub log_time_ns: u64,
    pub publish_time_ns: u64,
    pub payload: Vec<u8>,
}

// --- Writing ------------------------------------------------------------------

struct ChannelSlot {
    id: u16,
    sequence: u32,
}

/// Writes an MCAP bag. Call [`BagWriter::finish`] to write the summary and
/// footer; a dropped writer finishes on a best-effort basis, ignoring errors.
pub struct BagWriter {
    mcap: mcap::Writer<BufWriter<File>>,
    telemetry_schema: u16,
    command_schema: u16,
    channels: HashMap<String, ChannelSlot>,
}

impl BagWriter {
    /// Create (or truncate) the bag at `path`, embedding `scenario` (TOML) as
    /// the `astrum.scenario` metadata record when given.
    pub fn create(path: &Path, scenario: Option<&str>) -> Result<Self> {
        let file = BufWriter::new(File::create(path)?);
        // Chunks are zstd-compressed on the writing thread: no compressor pool.
        let options = mcap::WriteOptions::new().profile("astrum").compression_threads(0);
        let mut mcap = options.create(file)?;
        let telemetry_schema =
            mcap.add_schema(TELEMETRY_SCHEMA, ENCODING, SCHEMA_TEXT.as_bytes())?;
        let command_schema = mcap.add_schema(COMMAND_SCHEMA, ENCODING, SCHEMA_TEXT.as_bytes())?;
        if let Some(toml) = scenario {
            mcap.write_metadata(&Metadata {
                name: SCENARIO_METADATA.to_owned(),
                metadata: BTreeMap::from([(SCENARIO_KEY.to_owned(), toml.to_owned())]),
            })?;
        }
        Ok(Self { mcap, telemetry_schema, command_schema, channels: HashMap::new() })
    }

    /// Append one message on `topic` (`astrum/telemetry/<id>` or
    /// `astrum/command/<id>`; its channel is declared on first use).
    pub fn write(
        &mut self,
        topic: &str,
        log_time_ns: u64,
        publish_time_ns: u64,
        payload: &[u8],
    ) -> Result<()> {
        let slot = match self.channels.get_mut(topic) {
            Some(slot) => slot,
            None => {
                let schema = match TopicKind::of(topic) {
                    Some(TopicKind::Telemetry(_)) => self.telemetry_schema,
                    Some(TopicKind::Command(_)) => self.command_schema,
                    None => return Err(BagError::UnknownTopic(topic.to_owned())),
                };
                let id = self.mcap.add_channel(schema, topic, ENCODING, &BTreeMap::new())?;
                self.channels.entry(topic.to_owned()).or_insert(ChannelSlot { id, sequence: 0 })
            }
        };
        slot.sequence = slot.sequence.wrapping_add(1);
        let header = MessageHeader {
            channel_id: slot.id,
            sequence: slot.sequence,
            log_time: log_time_ns,
            publish_time: publish_time_ns,
        };
        self.mcap.write_to_known_channel(&header, payload)?;
        Ok(())
    }

    /// Write the summary section and footer, flush, and sync the file.
    pub fn finish(mut self) -> Result<()> {
        self.mcap.finish()?;
        let file = self.mcap.into_inner().into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        Ok(())
    }
}

// --- Reading ------------------------------------------------------------------

/// A complete MCAP bag loaded into memory.
pub struct BagReader {
    bytes: Vec<u8>,
    scenario: Option<String>,
}

impl BagReader {
    /// Load and validate the bag at `path`: a truncated or unfinished file (no
    /// footer / end magic) or malformed records are an error.
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut scenario = None;
        // A record-level scan (chunks are not decompressed) checks the framing
        // through the end magic and finds the metadata without trusting indexes.
        for record in mcap::read::LinearReader::new(&bytes)? {
            if let Record::Metadata(m) = record? {
                if m.name == SCENARIO_METADATA {
                    scenario = m.metadata.get(SCENARIO_KEY).cloned();
                }
            }
        }
        Ok(Self { bytes, scenario })
    }

    /// The scenario TOML recorded with the bag, if any.
    pub fn scenario(&self) -> Option<&str> {
        self.scenario.as_deref()
    }

    /// Every message, in log-time order (ties keep file order).
    pub fn messages(&self) -> Result<Vec<BagMessage>> {
        let mut out = Vec::new();
        for message in mcap::MessageStream::new(&self.bytes)? {
            let message = message?;
            out.push(BagMessage {
                topic: message.channel.topic.clone(),
                log_time_ns: message.log_time,
                publish_time_ns: message.publish_time,
                payload: message.data.into_owned(),
            });
        }
        out.sort_by_key(|m| m.log_time_ns);
        Ok(out)
    }
}

// --- Recording ----------------------------------------------------------------

/// Wall clock in ns since the Unix epoch (0 if the clock is before 1970).
pub fn wall_clock_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
}

/// Telemetry sim time in ns (`publish_time`).
fn sim_time_ns(sim_time_s: f64) -> u64 {
    // Saturating float->int cast: NaN and negative times become 0.
    (sim_time_s * 1e9).round() as u64
}

/// What a [`Recorder`] has written so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecordStats {
    /// Telemetry messages written.
    pub telemetry: u64,
    /// Command messages written.
    pub commands: u64,
    /// Telemetry samples dropped because their payload did not decode (no sim
    /// time to stamp them with).
    pub rejected: u64,
}

struct Received {
    topic: String,
    log_time_ns: u64,
    payload: Vec<u8>,
}

/// How long [`Recorder::run`] waits for a sample before re-checking `stop`.
const RECORD_POLL: Duration = Duration::from_millis(100);
/// How long [`Recorder::run`] waits for in-flight callbacks after unsubscribing.
const RECORD_DRAIN: Duration = Duration::from_millis(200);

/// Subscribes to every telemetry and command key and records them to a
/// [`BagWriter`].
///
/// Zenoh's receive callback only timestamps and copies each sample into an
/// unbounded channel, so recording never blocks the transport on disk I/O and
/// never drops a sample it received; [`Recorder::run`] does the writing.
pub struct Recorder {
    rx: Receiver<Received>,
    subscribers: Vec<Subscriber<()>>,
}

impl Recorder {
    /// Declare the subscriptions. Samples published after this returns are
    /// captured (buffered until [`Recorder::run`] writes them).
    pub fn subscribe(session: &Session) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let mut subscribers = Vec::with_capacity(2);
        for key in [keys::TELEMETRY_ALL, keys::COMMAND_ALL] {
            let tx = tx.clone();
            let subscriber = session
                .raw()
                .declare_subscriber(key)
                .callback(move |sample| {
                    let received = Received {
                        log_time_ns: wall_clock_ns(),
                        topic: sample.key_expr().as_str().to_owned(),
                        payload: sample.payload().to_bytes().into_owned(),
                    };
                    // Fails only once `run` has returned and dropped the receiver.
                    let _ = tx.send(received);
                })
                .wait()
                .map_err(|e| NetError::Zenoh(e.to_string()))?;
            subscribers.push(subscriber);
        }
        Ok(Self { rx, subscribers })
    }

    /// Write samples to `bag` until `stop(&stats)` returns true (checked after
    /// every sample and at least every 100 ms), then unsubscribe and write the
    /// samples still in flight. The caller finishes the bag.
    pub fn run(
        self,
        bag: &mut BagWriter,
        mut stop: impl FnMut(&RecordStats) -> bool,
    ) -> Result<RecordStats> {
        let Recorder { rx, subscribers } = self;
        let mut stats = RecordStats::default();
        while !stop(&stats) {
            match rx.recv_timeout(RECORD_POLL) {
                Ok(received) => record_one(bag, &mut stats, received)?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        for subscriber in subscribers {
            subscriber.undeclare().wait().map_err(|e| NetError::Zenoh(e.to_string()))?;
        }
        // The callbacks own the senders: the channel disconnects once Zenoh has
        // dropped them; the timeout bounds the wait should it hold on to one.
        while let Ok(received) = rx.recv_timeout(RECORD_DRAIN) {
            record_one(bag, &mut stats, received)?;
        }
        Ok(stats)
    }
}

fn record_one(bag: &mut BagWriter, stats: &mut RecordStats, received: Received) -> Result<()> {
    let Received { topic, log_time_ns, payload } = received;
    match TopicKind::of(&topic) {
        Some(TopicKind::Telemetry(_)) => match astrum_proto::decode_telemetry(&payload) {
            Ok(t) => {
                bag.write(&topic, log_time_ns, sim_time_ns(t.sim_time_s), &payload)?;
                stats.telemetry += 1;
            }
            Err(_) => stats.rejected += 1,
        },
        Some(TopicKind::Command(_)) => {
            bag.write(&topic, log_time_ns, log_time_ns, &payload)?;
            stats.commands += 1;
        }
        // The subscriptions only match the two streams; a non-numeric id lands here.
        None => stats.rejected += 1,
    }
    Ok(())
}

// --- Replay -------------------------------------------------------------------

/// When to publish each message, as offsets from the start of playback, given
/// their `publish_time`s (sim time, ns) in playback order.
///
/// Each gap to the previous message is divided by `rate` (2.0 = twice real
/// time) and capped at [`MAX_REPLAY_GAP`]; time running backwards (a sim
/// restart) is a zero gap. `rate <= 0` (or NaN) means unpaced: every offset is
/// zero.
pub fn replay_schedule(publish_times_ns: &[u64], rate: f64) -> Vec<Duration> {
    let paced = rate > 0.0;
    let mut offset = Duration::ZERO;
    let mut prev: Option<u64> = None;
    publish_times_ns
        .iter()
        .map(|&t| {
            if let (true, Some(p)) = (paced, prev) {
                let gap_s = t.saturating_sub(p) as f64 / 1e9 / rate;
                offset += Duration::try_from_secs_f64(gap_s)
                    .map_or(MAX_REPLAY_GAP, |d| d.min(MAX_REPLAY_GAP));
            }
            prev = Some(t);
            offset
        })
        .collect()
}

/// What [`replay`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayStats {
    /// Telemetry messages published.
    pub telemetry: u64,
    /// Command messages skipped (recorded for analysis, never replayed).
    pub commands: u64,
    /// Messages on any other topic, skipped.
    pub skipped: u64,
}

/// Publish the bag's telemetry on `session` exactly as the sim did (same keys,
/// same payload bytes, the sim's QoS), paced by [`replay_schedule`]. Commands
/// are not replayed.
pub fn replay(session: &Session, bag: &BagReader, rate: f64) -> Result<ReplayStats> {
    let mut stats = ReplayStats::default();
    let mut telemetry = Vec::new();
    for message in bag.messages()? {
        match TopicKind::of(&message.topic) {
            Some(TopicKind::Telemetry(id)) => telemetry.push((id, message)),
            Some(TopicKind::Command(_)) => stats.commands += 1,
            None => stats.skipped += 1,
        }
    }
    let times: Vec<u64> = telemetry.iter().map(|(_, m)| m.publish_time_ns).collect();
    let schedule = replay_schedule(&times, rate);

    let mut publisher = TelemetryPublisher::new(session);
    let start = Instant::now();
    for ((id, message), offset) in telemetry.into_iter().zip(schedule) {
        let wait = (start + offset).saturating_duration_since(Instant::now());
        if !wait.is_zero() {
            std::thread::sleep(wait);
        }
        publisher.publish_raw(id, message.payload)?;
        stats.telemetry += 1;
    }
    Ok(stats)
}
