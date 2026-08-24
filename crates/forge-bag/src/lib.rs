//! `forge-bag` — record the FORGE telemetry stream to a file and replay it.
//!
//! A bag is a flat sequence of `BagFrame` Cap'n Proto messages (see
//! `forge-proto`): each is a wall-clock receive timestamp plus one
//! [`TelemetrySample`]. Recording and replay are deliberately *just another ZMQ
//! client* — the recorder is a `SUB`, and replay is a `PUB` — so a replayed bag
//! reaches any subscriber (a controller, the future replay renderer) byte-for-
//! byte identically to the live sim.
//!
//! This library provides the [`BagWriter`] append helper; reading is
//! [`forge_proto::BagReader`], re-exported here as [`BagReader`].

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use forge_proto::{write_bag_frame, TelemetrySample};

pub use forge_proto::BagReader;

/// Appends `(recv_wall_ns, sample)` frames to a bag stream.
pub struct BagWriter<W: Write> {
    writer: W,
    frames: u64,
}

impl BagWriter<BufWriter<File>> {
    /// Create (or truncate) a bag file at `path`.
    pub fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self::new(BufWriter::new(File::create(path)?)))
    }
}

impl<W: Write> BagWriter<W> {
    /// Wrap an arbitrary writer (file, buffer, socket).
    pub fn new(writer: W) -> Self {
        Self { writer, frames: 0 }
    }

    /// Append one frame: the wall-clock receive time (ns since the Unix epoch)
    /// and the telemetry sample.
    pub fn write_frame(
        &mut self,
        recv_wall_ns: u64,
        sample: &TelemetrySample,
    ) -> capnp::Result<()> {
        write_bag_frame(&mut self.writer, recv_wall_ns, sample)?;
        self.frames += 1;
        Ok(())
    }

    /// Flush buffered bytes to the underlying writer.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }

    /// Number of frames written so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }
}
