#include "astrum/log.hpp"

#include <cstdio>
#include <utility>

namespace astrum {

std::string_view to_string(LogLevel level) {
  switch (level) {
    case LogLevel::Debug:
      return "debug";
    case LogLevel::Info:
      return "info";
    case LogLevel::Warning:
      return "warning";
    case LogLevel::Error:
      return "error";
  }
  return "unknown";
}

Logger stderr_logger(std::string tag) {
  return [tag = std::move(tag)](LogLevel level, std::string_view message) {
    if (level == LogLevel::Debug) return;
    std::string line = "[" + tag + "] ";
    if (level == LogLevel::Warning || level == LogLevel::Error) {
      line += to_string(level);
      line += ": ";
    }
    line += message;
    line += '\n';
    // One fwrite per line: lines from several threads do not interleave.
    std::fwrite(line.data(), 1, line.size(), stderr);
  };
}

}  // namespace astrum
