//! `forge-net` — the FORGE ZMQ transport layer.
//!
//! Two channels, both carrying [`forge_proto`] payloads on the wire:
//!
//! * **Telemetry** — sim `PUB` → subscribers `SUB`. Each message is two frames:
//!   a fixed 4-byte big-endian aircraft-id topic followed by the encoded
//!   [`TelemetrySample`]. Subscribers filter per aircraft with [`TelemetrySubscriber::subscribe`]
//!   or take the whole fleet with [`TelemetrySubscriber::subscribe_all`]. A fixed-width
//!   topic avoids ZMQ prefix-match collisions (id `1` never matches `10`/`123`).
//! * **Commands** — controllers `PUSH` → sim `PULL` (fan-in). One encoded
//!   [`CommandMsg`] per message; the sim drains them non-blockingly each tick.
//!
//! The bag recorder is just another `SUB` client, and replay is just another
//! `PUB` — replay feeds any subscriber (including the future replay renderer)
//! byte-for-byte identically to the live sim.
//!
//! `zmq::Socket` is `!Send`, so these wrappers are `!Send` too; hold them as
//! Bevy `NonSend` resources (fine under `ScheduleRunnerPlugin`, which keeps the
//! app on one thread).

use forge_proto::{
    decode_command, decode_telemetry, encode_command, encode_telemetry, CommandMsg, TelemetrySample,
};

// Re-export so downstream crates share one libzmq / one Context type.
pub use zmq;

/// Default endpoint the sim binds for publishing telemetry.
pub const DEFAULT_TELEMETRY_ENDPOINT: &str = "tcp://0.0.0.0:5555";
/// Default endpoint the sim binds for receiving commands.
pub const DEFAULT_COMMAND_ENDPOINT: &str = "tcp://0.0.0.0:5556";

/// The 4-byte topic frame used for an aircraft's telemetry.
#[inline]
pub fn topic_for(aircraft_id: u32) -> [u8; 4] {
    aircraft_id.to_be_bytes()
}

/// Create a fresh ZMQ context. Share one context across sockets that talk over
/// `inproc://`; TCP sockets may each have their own.
pub fn context() -> zmq::Context {
    zmq::Context::new()
}

/// Sim-side telemetry broadcaster (ZMQ `PUB`).
pub struct TelemetryPublisher {
    socket: zmq::Socket,
}

impl TelemetryPublisher {
    /// Bind a `PUB` socket to `endpoint` (e.g. [`DEFAULT_TELEMETRY_ENDPOINT`]).
    pub fn bind(ctx: &zmq::Context, endpoint: &str) -> zmq::Result<Self> {
        let socket = ctx.socket(zmq::PUB)?;
        socket.bind(endpoint)?;
        Ok(Self { socket })
    }

    /// Publish one sample as `[topic, payload]`. The topic is the aircraft id
    /// (from `sample.aircraft_id`) as 4 big-endian bytes.
    pub fn publish(&self, sample: &TelemetrySample) -> zmq::Result<()> {
        let topic = topic_for(sample.aircraft_id);
        let payload = encode_telemetry(sample);
        self.socket.send_multipart([&topic[..], &payload[..]], 0)
    }
}

/// Subscriber-side telemetry reader (ZMQ `SUB`) — used by the bag recorder,
/// controllers, and the future replay renderer.
pub struct TelemetrySubscriber {
    socket: zmq::Socket,
}

impl TelemetrySubscriber {
    /// Connect a `SUB` socket to a publisher `endpoint`. Nothing is delivered
    /// until you call [`subscribe`](Self::subscribe) or
    /// [`subscribe_all`](Self::subscribe_all).
    pub fn connect(ctx: &zmq::Context, endpoint: &str) -> zmq::Result<Self> {
        let socket = ctx.socket(zmq::SUB)?;
        socket.connect(endpoint)?;
        Ok(Self { socket })
    }

    /// Subscribe to a single aircraft's telemetry.
    pub fn subscribe(&self, aircraft_id: u32) -> zmq::Result<()> {
        self.socket.set_subscribe(&topic_for(aircraft_id))
    }

    /// Subscribe to every aircraft (empty prefix matches all topics).
    pub fn subscribe_all(&self) -> zmq::Result<()> {
        self.socket.set_subscribe(b"")
    }

    /// Set a receive timeout in milliseconds; `-1` blocks forever (the default).
    pub fn set_recv_timeout_ms(&self, ms: i32) -> zmq::Result<()> {
        self.socket.set_rcvtimeo(ms)
    }

    /// Block for the next sample. On a configured timeout this returns
    /// `Err(zmq::Error::EAGAIN)`.
    pub fn recv(&self) -> zmq::Result<TelemetrySample> {
        let parts = self.socket.recv_multipart(0)?;
        Self::decode(&parts)
    }

    /// Non-blocking receive: `Ok(None)` when no message is queued.
    pub fn try_recv(&self) -> zmq::Result<Option<TelemetrySample>> {
        match self.socket.recv_multipart(zmq::DONTWAIT) {
            Ok(parts) => Ok(Some(Self::decode(&parts)?)),
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn decode(parts: &[Vec<u8>]) -> zmq::Result<TelemetrySample> {
        // Payload is the last frame; a lone frame means no topic was sent.
        let payload = parts.last().ok_or(zmq::Error::EPROTO)?;
        decode_telemetry(payload).map_err(|_| zmq::Error::EPROTO)
    }
}

/// Sim-side command intake (ZMQ `PULL`, fan-in from every controller).
pub struct CommandReceiver {
    socket: zmq::Socket,
}

impl CommandReceiver {
    /// Bind a `PULL` socket to `endpoint` (e.g. [`DEFAULT_COMMAND_ENDPOINT`]).
    pub fn bind(ctx: &zmq::Context, endpoint: &str) -> zmq::Result<Self> {
        let socket = ctx.socket(zmq::PULL)?;
        socket.bind(endpoint)?;
        Ok(Self { socket })
    }

    /// Drain every queued command without blocking. Undecodable messages are
    /// skipped so one malformed frame can't wedge the intake.
    pub fn drain(&self) -> Vec<CommandMsg> {
        let mut out = Vec::new();
        // Stops on EAGAIN (queue empty) or any error.
        while let Ok(bytes) = self.socket.recv_bytes(zmq::DONTWAIT) {
            if let Ok(cmd) = decode_command(&bytes) {
                out.push(cmd);
            }
        }
        out
    }
}

/// Controller-side command sender (ZMQ `PUSH`).
pub struct CommandSender {
    socket: zmq::Socket,
}

impl CommandSender {
    /// Connect a `PUSH` socket to the sim's command `endpoint`.
    pub fn connect(ctx: &zmq::Context, endpoint: &str) -> zmq::Result<Self> {
        let socket = ctx.socket(zmq::PUSH)?;
        socket.connect(endpoint)?;
        Ok(Self { socket })
    }

    /// Send one command.
    pub fn send(&self, cmd: &CommandMsg) -> zmq::Result<()> {
        self.socket.send(encode_command(cmd), 0)
    }
}
