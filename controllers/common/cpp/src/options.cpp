#include "astrum/options.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace astrum {
namespace {

std::optional<double> parse_double(const std::string& text) {
  if (text.empty()) return std::nullopt;
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(text.c_str(), &end);
  if (errno != 0 || end != text.c_str() + text.size() || !std::isfinite(v)) return std::nullopt;
  return v;
}

std::optional<long long> parse_int(const std::string& text) {
  if (text.empty()) return std::nullopt;
  char* end = nullptr;
  errno = 0;
  const long long v = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end != text.c_str() + text.size()) return std::nullopt;
  return v;
}

std::string format_double(double v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

}  // namespace

Options::Options(std::string program, std::string summary)
    : program_(std::move(program)), summary_(std::move(summary)) {}

Options& Options::add_option(std::string flag, std::string meta, std::string help, const char* env,
                             std::string default_text,
                             std::function<bool(const std::string&)> set) {
  for (const Option& opt : options_) {
    if (opt.flag == flag) throw std::logic_error("option " + flag + " added twice");
  }
  options_.push_back(Option{std::move(flag), std::move(meta), std::move(help),
                            env != nullptr ? env : "", std::move(default_text), std::move(set),
                            false});
  return *this;
}

Options& Options::add(std::string flag, std::string meta, std::string help, const char* env,
                      std::string* target) {
  return add_option(std::move(flag), std::move(meta), std::move(help), env, *target,
                    [target](const std::string& v) {
                      *target = v;
                      return true;
                    });
}

Options& Options::add(std::string flag, std::string meta, std::string help, const char* env,
                      double* target) {
  return add_option(std::move(flag), std::move(meta), std::move(help), env, format_double(*target),
                    [target](const std::string& v) {
                      const auto d = parse_double(v);
                      if (d) *target = *d;
                      return d.has_value();
                    });
}

Options& Options::add(std::string flag, std::string meta, std::string help, const char* env,
                      int* target) {
  return add_option(
      std::move(flag), std::move(meta), std::move(help), env, std::to_string(*target),
      [target](const std::string& v) {
        const auto i = parse_int(v);
        if (!i || *i < std::numeric_limits<int>::min() || *i > std::numeric_limits<int>::max()) {
          return false;
        }
        *target = static_cast<int>(*i);
        return true;
      });
}

Options& Options::add(std::string flag, std::string meta, std::string help, const char* env,
                      std::uint32_t* target) {
  return add_option(std::move(flag), std::move(meta), std::move(help), env, std::to_string(*target),
                    [target](const std::string& v) {
                      const auto i = parse_int(v);
                      if (!i || *i < 0 || *i > std::numeric_limits<std::uint32_t>::max()) {
                        return false;
                      }
                      *target = static_cast<std::uint32_t>(*i);
                      return true;
                    });
}

Options& Options::required() {
  if (options_.empty()) throw std::logic_error("Options::required() before any add()");
  options_.back().required = true;
  return *this;
}

std::optional<int> Options::parse(int argc, const char* const* argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout << usage();
      return 0;
    }
  }

  std::vector<bool> given(options_.size(), false);
  for (std::size_t k = 0; k < options_.size(); ++k) {
    const Option& opt = options_[k];
    if (opt.env.empty()) continue;
    const char* value = std::getenv(opt.env.c_str());
    if (value == nullptr) continue;
    if (!opt.set(value)) {
      std::cerr << program_ << ": invalid value '" << value << "' in $" << opt.env << "\n";
      return 2;
    }
    given[k] = true;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    std::string key = arg;
    std::optional<std::string> value;
    if (const auto eq = arg.find('='); arg.rfind("--", 0) == 0 && eq != std::string::npos) {
      key = arg.substr(0, eq);
      value = arg.substr(eq + 1);
    }
    std::size_t match = options_.size();
    for (std::size_t k = 0; k < options_.size(); ++k) {
      if (options_[k].flag == key) match = k;
    }
    if (match == options_.size()) {
      std::cerr << program_ << ": unknown argument '" << arg << "'\n" << usage();
      return 2;
    }
    if (!value) {
      if (i + 1 >= argc) {
        std::cerr << program_ << ": missing value for " << key << "\n";
        return 2;
      }
      value = argv[++i];
    }
    if (!options_[match].set(*value)) {
      std::cerr << program_ << ": invalid value '" << *value << "' for " << key << "\n";
      return 2;
    }
    given[match] = true;
  }

  for (std::size_t k = 0; k < options_.size(); ++k) {
    const Option& opt = options_[k];
    if (opt.required && !given[k]) {
      std::cerr << program_ << ": " << opt.flag << " is required"
                << (opt.env.empty() ? "" : " (or $" + opt.env + ")") << "\n";
      return 2;
    }
  }
  return std::nullopt;
}

std::string Options::usage() const {
  std::ostringstream os;
  os << "Usage: " << program_ << " [options]\n\n" << summary_ << "\n\nOptions:\n";
  for (const Option& opt : options_) {
    std::string lhs = "  " + opt.flag + " <" + opt.meta + ">";
    if (lhs.size() < 28) lhs.resize(28, ' ');
    os << lhs << " " << opt.help << (opt.required ? " [required" : " [default " + opt.default_text);
    if (!opt.env.empty()) os << ", env " << opt.env;
    os << "]\n";
  }
  os << "  -h, --help                 show this help and exit\n"
     << "Precedence: CLI flag > environment variable > default.\n";
  return os.str();
}

}  // namespace astrum
