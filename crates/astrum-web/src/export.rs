//! Bag → [`ExportDoc`]: decode every telemetry and event message of an MCAP bag
//! into the renderer's frame and event shapes.

use std::path::Path;

use astrum_bag::{BagError, BagReader, TopicKind};
use astrum_proto::{decode_event, decode_telemetry};

use crate::{EventJson, ExportDoc, ExportFrame, FrameJson};

/// Read the bag at `path` and bake its telemetry and events into an
/// [`ExportDoc`].
///
/// Other topics (commands) are ignored. A telemetry payload that fails to
/// decode, or whose aircraft id disagrees with its topic, and an event payload
/// that fails to decode (including a kind from a newer schema) are untrusted
/// input: each is counted in [`crate::ExportMeta::rejected`] and skipped.
pub fn export_bag(path: &Path) -> Result<ExportDoc, BagError> {
    let reader = BagReader::open(path)?;
    let mut frames = Vec::new();
    let mut events = Vec::new();
    let mut rejected = 0u64;
    for msg in reader.messages()? {
        match TopicKind::of(&msg.topic) {
            Some(TopicKind::Telemetry(topic_id)) => match decode_telemetry(&msg.payload) {
                Ok(t) if t.aircraft_id == topic_id => frames.push(ExportFrame {
                    frame: FrameJson::from_telemetry(&t),
                    log_time_ns: msg.log_time_ns,
                }),
                _ => rejected += 1,
            },
            Some(TopicKind::Events) => match decode_event(&msg.payload) {
                Ok(e) => events.push(EventJson::from_event(&e)),
                Err(_) => rejected += 1,
            },
            Some(TopicKind::Command(_)) | None => {}
        }
    }
    Ok(ExportDoc::new(frames, events, reader.scenario().map(str::to_owned), rejected))
}
