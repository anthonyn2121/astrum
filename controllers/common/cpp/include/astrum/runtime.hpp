// The law runtime (astrum::runtime): a controller program is a table of laws handed to
// run_main (controllers/README.md, "Writing a controller").
//
//   #include "astrum/runtime.hpp"
//   int main(int argc, char** argv) {
//     const astrum::LawEntry laws[] = {{"hold", "hold the reference", nullptr,
//         [](const astrum::LawContext& ctx) { return std::make_unique<Hold>(ctx.reference); }}};
//     return astrum::run_main(argc, argv, laws, "hold", "...");
//   }
//
// The Law interface, StepInput, Stop and LawResult are declared in astrum/loop.hpp
// (the loop runs them); this header brings them in.

#ifndef ASTRUM_RUNTIME_HPP
#define ASTRUM_RUNTIME_HPP

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "astrum/cli.hpp"
#include "astrum/loop.hpp"
#include "astrum/options.hpp"
#include "astrum/types.hpp"

namespace astrum {

// The shared flags a law is built from.
struct LawContext {
  Reference reference;    // --heading / --alt / --speed
  AttitudeLimits limits;  // --max-bank-deg / --max-pitch-deg
};

struct LawEntry {
  std::string name;     // --law <name>
  std::string summary;  // one line for --help
  // Registers the law's own flags (may be empty). Every entry's flags are registered,
  // whichever law runs.
  std::function<void(Options&)> add_options;
  // Builds the law after the flags are parsed. Validate here: an exception is a usage
  // error (exit 2).
  std::function<std::unique_ptr<Law>(const LawContext&)> make;
};

// Parse the shared flags (controllers/README.md, "Flags") plus every law's own, build
// the selected law (--law, default the first entry), open the Zenoh session, subscribe,
// and run the control loop for --id. Returns the exit code: 0 after --help, 2 on a
// usage error, 1 when the sim is unreachable before the startup deadline, otherwise
// exit_code() of the loop's Outcome.
int run_main(int argc, char** argv, std::span<const LawEntry> laws, std::string_view program,
             std::string_view summary);

}  // namespace astrum

#endif  // ASTRUM_RUNTIME_HPP
