//! In-process recording (`astrum-sim --record`): every tick's telemetry, every
//! command the sim decodes, and every event, written to an MCAP bag in the
//! `astrum-bag` format, so `astrum-bag-play`, `astrum-web-export` and the
//! viewer read it.
//!
//! Unlike `astrum-bag-record`, which subscribes over Zenoh, this cannot drop a
//! sample and adds no network hop to the tick — what an unpaced run needs. Chunk
//! compression and disk I/O happen on a writer thread behind a bounded queue: a
//! disk slower than the sim slows the sim down, it never loses data.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;

use astrum_bag::{sim_time_ns, BagError, BagWriter, RecordStats, TopicKind};
use astrum_net::keys;
use bevy::prelude::Resource;

/// Messages queued between the sim and the writer thread (~20 MB of telemetry).
const QUEUE: usize = 1 << 16;

enum Entry {
    Telemetry { id: u32, log_time_ns: u64, publish_time_ns: u64, payload: Vec<u8> },
    Command { id: u32, log_time_ns: u64, payload: Vec<u8> },
    Event { log_time_ns: u64, publish_time_ns: u64, payload: Vec<u8> },
}

/// The sim's end of a recording: queues encoded messages for the writer.
#[derive(Resource)]
pub struct TickRecorder {
    tx: SyncSender<Entry>,
}

impl TickRecorder {
    /// Queue one encoded `Telemetry`. `false` once the writer has stopped (it
    /// failed; [`Recording::finish`] reports why).
    pub fn telemetry(&self, id: u32, log_time_ns: u64, sim_time_s: f64, payload: Vec<u8>) -> bool {
        let publish_time_ns = sim_time_ns(sim_time_s);
        self.tx.send(Entry::Telemetry { id, log_time_ns, publish_time_ns, payload }).is_ok()
    }

    /// Queue one encoded `Command` (stamped with its receipt time). `false` once
    /// the writer has stopped.
    pub fn command(&self, id: u32, log_time_ns: u64, payload: Vec<u8>) -> bool {
        self.tx.send(Entry::Command { id, log_time_ns, payload }).is_ok()
    }

    /// Queue one encoded `Event`. `false` once the writer has stopped.
    pub fn event(&self, log_time_ns: u64, sim_time_s: f64, payload: Vec<u8>) -> bool {
        let publish_time_ns = sim_time_ns(sim_time_s);
        self.tx.send(Entry::Event { log_time_ns, publish_time_ns, payload }).is_ok()
    }
}

/// The writer's end: [`Recording::finish`] once the app holding the
/// [`TickRecorder`] is gone (that closes the queue).
pub struct Recording {
    path: PathBuf,
    writer: JoinHandle<Result<RecordStats, BagError>>,
}

impl Recording {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Wait for the writer to drain the queue, then write the bag's summary and
    /// footer. Blocks until every [`TickRecorder`] has been dropped.
    pub fn finish(self) -> Result<RecordStats, BagError> {
        self.writer.join().unwrap_or_else(|_| {
            Err(BagError::Io(std::io::Error::other("the recording thread panicked")))
        })
    }
}

/// Create the bag at `path` (with the scenario as its metadata) and start the
/// writer thread. File errors surface here, before the sim runs.
pub fn start(path: &Path, scenario_toml: &str) -> Result<(TickRecorder, Recording), BagError> {
    let bag = BagWriter::create(path, Some(scenario_toml))?;
    let (tx, rx) = sync_channel(QUEUE);
    let writer = std::thread::Builder::new()
        .name("astrum-record".into())
        .spawn(move || write_all(bag, rx))?;
    Ok((TickRecorder { tx }, Recording { path: path.to_owned(), writer }))
}

fn write_all(mut bag: BagWriter, rx: Receiver<Entry>) -> Result<RecordStats, BagError> {
    let mut topics: HashMap<TopicKind, String> = HashMap::new();
    let mut stats = RecordStats::default();
    // Ends when every sender is gone; an error drops `rx`, so the sim's next
    // send fails and it stops.
    for entry in rx {
        match entry {
            Entry::Telemetry { id, log_time_ns, publish_time_ns, payload } => {
                let topic =
                    topics.entry(TopicKind::Telemetry(id)).or_insert_with(|| keys::telemetry(id));
                bag.write(topic, log_time_ns, publish_time_ns, &payload)?;
                stats.telemetry += 1;
            }
            Entry::Command { id, log_time_ns, payload } => {
                let topic =
                    topics.entry(TopicKind::Command(id)).or_insert_with(|| keys::command(id));
                bag.write(topic, log_time_ns, log_time_ns, &payload)?;
                stats.commands += 1;
            }
            Entry::Event { log_time_ns, publish_time_ns, payload } => {
                bag.write(keys::EVENTS, log_time_ns, publish_time_ns, &payload)?;
                stats.events += 1;
            }
        }
    }
    bag.finish()?;
    Ok(stats)
}
