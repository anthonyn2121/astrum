// The log sink of the transport and the control loop (part of astrum::client): an
// injected function instead of a hard-coded std::cerr, so a host program routes the
// lines wherever it logs and tests capture them.

#ifndef ASTRUM_LOG_HPP
#define ASTRUM_LOG_HPP

#include <functional>
#include <string>
#include <string_view>

namespace astrum {

enum class LogLevel { Debug, Info, Warning, Error };

// "debug" / "info" / "warning" / "error".
std::string_view to_string(LogLevel level);

using Logger = std::function<void(LogLevel, std::string_view)>;

// Writes "[<tag>] <message>" lines (warnings and errors flagged) to stderr; Debug lines
// are dropped. Thread-safe: one line per call.
Logger stderr_logger(std::string tag);

}  // namespace astrum

#endif  // ASTRUM_LOG_HPP
