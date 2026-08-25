// Law registry for the ASTRUM optimal-control controller (C++).
//
// Each law in laws/<name>.cpp exposes one factory function declared here; the
// runtime (runtime.cpp) maps the `--law <name>` string to the matching factory.
// To add a law: write laws/mylaw.cpp with a `class MyLaw : public Law`, declare
// `std::unique_ptr<Law> make_mylaw_law(const Cli&)` below, add it to the table in
// runtime.cpp's make_law(), and list its source in CMakeLists.txt.

#ifndef ASTRUM_LAWS_HPP
#define ASTRUM_LAWS_HPP

#include <memory>

#include "astrum_control.hpp"

namespace astrum {

std::unique_ptr<Law> make_lqr_law(const Cli& cli);
std::unique_ptr<Law> make_mpc_law(const Cli& cli);

}  // namespace astrum

#endif  // ASTRUM_LAWS_HPP
