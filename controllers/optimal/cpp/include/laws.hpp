// Law registry for the ASTRUM optimal-control controller (C++).
//
// Each law in laws/<name>.cpp exposes one factory declared here; make_law() in
// src/runtime.cpp maps `--law <name>` to it. To add a law: write laws/mylaw.cpp with a
// `class MyLaw : public Law`, declare `std::unique_ptr<Law> make_mylaw_law(const LawConfig&)`
// below, add it to make_law(), and list its source in CMakeLists.txt.

#ifndef ASTRUM_LAWS_HPP
#define ASTRUM_LAWS_HPP

#include <memory>

#include "astrum_control.hpp"

namespace astrum {

// Both throw std::runtime_error if their gain synthesis fails (e.g. dlqr non-convergence)
// and std::invalid_argument for unusable configuration.
std::unique_ptr<Law> make_lqr_law(const LawConfig& config);
std::unique_ptr<Law> make_mpc_law(const LawConfig& config);

}  // namespace astrum

#endif  // ASTRUM_LAWS_HPP
