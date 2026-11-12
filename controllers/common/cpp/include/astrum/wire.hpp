// Cap'n Proto codec for the ASTRUM wire schema v3 (generated types: namespace
// astrum::wire, from crates/astrum-proto/schema/astrum.capnp). One unpacked flat message
// per Zenoh payload. Decoding treats its input as untrusted: any malformed payload —
// including an unknown Command.control or Event kind from a newer schema — yields an
// error value, never an exception.

#ifndef ASTRUM_WIRE_HPP
#define ASTRUM_WIRE_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "astrum/types.hpp"

namespace astrum {

// A decode result: a value, or the reason the payload was rejected.
template <class T>
struct Decoded {
  std::optional<T> value;
  std::string error;  // empty when value is set

  explicit operator bool() const { return value.has_value(); }
};

// Decode one payload (any alignment). Rejects payloads that are not a whole number of
// words, malformed messages, and trailing bytes after the message.
Decoded<VehicleState> decode_telemetry(std::span<const std::uint8_t> bytes) noexcept;
Decoded<Command> decode_command(std::span<const std::uint8_t> bytes) noexcept;
Decoded<Event> decode_event(std::span<const std::uint8_t> bytes) noexcept;
Decoded<TickEnd> decode_tick_end(std::span<const std::uint8_t> bytes) noexcept;
Decoded<RaceCourse> decode_race_course(std::span<const std::uint8_t> bytes) noexcept;
Decoded<RaceState> decode_race_state(std::span<const std::uint8_t> bytes) noexcept;

// Encode as a flat Cap'n Proto message (the bytes that go in a Zenoh payload).
std::vector<std::uint8_t> encode_telemetry(const VehicleState& state);
std::vector<std::uint8_t> encode_command(const Command& command);
std::vector<std::uint8_t> encode_event(const Event& event);
std::vector<std::uint8_t> encode_tick_end(const TickEnd& tick_end);
std::vector<std::uint8_t> encode_race_course(const RaceCourse& course);
std::vector<std::uint8_t> encode_race_state(const RaceState& state);

}  // namespace astrum

#endif  // ASTRUM_WIRE_HPP
