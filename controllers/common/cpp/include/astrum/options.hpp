// Command-line options with ASTRUM_* environment fallbacks (part of astrum::cli).
//
// Precedence, as everywhere in ASTRUM: CLI flag > environment variable > built-in
// default (the target's value when the option is added). Accepts `--flag value` and
// `--flag=value`; numbers must parse completely and be finite. The shared controller
// flags are added by the helpers in astrum/cli.hpp.

#ifndef ASTRUM_OPTIONS_HPP
#define ASTRUM_OPTIONS_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace astrum {

class Options {
 public:
  Options(std::string program, std::string summary);

  // `env` may be nullptr (no environment fallback).
  Options& add(std::string flag, std::string meta, std::string help, const char* env,
               std::string* target);
  Options& add(std::string flag, std::string meta, std::string help, const char* env,
               double* target);
  Options& add(std::string flag, std::string meta, std::string help, const char* env, int* target);
  Options& add(std::string flag, std::string meta, std::string help, const char* env,
               std::uint32_t* target);

  // Mark the option added last as required: parse() fails unless the flag or its
  // environment variable sets it.
  Options& required();

  // Apply the environment, then argv. Returns std::nullopt to continue, or the exit code
  // the program should return now: 0 after --help (usage on stdout), 2 on a usage error
  // (reported on stderr).
  std::optional<int> parse(int argc, const char* const* argv);

  std::string usage() const;

 private:
  struct Option {
    std::string flag, meta, help, env, default_text;
    std::function<bool(const std::string&)> set;  // false: value rejected
    bool required = false;
  };
  Options& add_option(std::string flag, std::string meta, std::string help, const char* env,
                      std::string default_text, std::function<bool(const std::string&)> set);

  std::string program_;
  std::string summary_;
  std::vector<Option> options_;
};

}  // namespace astrum

#endif  // ASTRUM_OPTIONS_HPP
