// Cap'n Proto codec for the ASTRUM wire schema (types: namespace astrum::wire, generated
// from crates/astrum-proto/schema/astrum.capnp). One unpacked flat message per Zenoh
// payload. Decoding treats its input as untrusted: any malformed payload yields an
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

// Decode a Telemetry payload (any alignment). Rejects payloads that are not a whole
// number of words, malformed messages, and trailing bytes after the message.
Decoded<VehicleState> decode_telemetry(std::span<const std::uint8_t> bytes) noexcept;

// Decode a Command payload (same rules as decode_telemetry).
Decoded<CommandMessage> decode_command(std::span<const std::uint8_t> bytes) noexcept;

// Encode as a flat Cap'n Proto message (the bytes that go in a Zenoh payload).
std::vector<std::uint8_t> encode_telemetry(const VehicleState& state);
std::vector<std::uint8_t> encode_command(const CommandMessage& message);

}  // namespace astrum

#endif  // ASTRUM_WIRE_HPP
