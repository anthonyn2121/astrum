//! Bag → [`ExportDoc`]: decode every telemetry message of an MCAP bag into the
//! renderer's frame shape.

use std::path::Path;

use astrum_bag::{BagError, BagReader};
use astrum_net::keys;
use astrum_proto::decode_telemetry;

use crate::{ExportDoc, ExportFrame, FrameJson};

/// Read the bag at `path` and bake its telemetry into an [`ExportDoc`].
///
/// Non-telemetry topics (commands) are ignored. A telemetry payload that fails
/// to decode, or whose aircraft id disagrees with its topic, is untrusted input:
/// it is counted in [`crate::ExportMeta::rejected`] and skipped.
pub fn export_bag(path: &Path) -> Result<ExportDoc, BagError> {
    let reader = BagReader::open(path)?;
    let mut frames = Vec::new();
    let mut rejected = 0u64;
    for msg in reader.messages()? {
        let Some(topic_id) = keys::id_of(&msg.topic, keys::TELEMETRY_PREFIX) else { continue };
        match decode_telemetry(&msg.payload) {
            Ok(t) if t.aircraft_id == topic_id => frames.push(ExportFrame {
                frame: FrameJson::from_telemetry(&t),
                log_time_ns: msg.log_time_ns,
            }),
            _ => rejected += 1,
        }
    }
    Ok(ExportDoc::new(frames, reader.scenario().map(str::to_owned), rejected))
}
