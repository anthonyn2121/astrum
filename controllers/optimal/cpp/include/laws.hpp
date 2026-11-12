// The laws of the ASTRUM optimal-control controller (C++), on the shared runtime
// (astrum/runtime.hpp). Each laws/<name>.cpp exposes a factory and a LawEntry;
// src/main.cpp hands the entries to astrum::run_main. To add a law: write
// laws/mylaw.cpp with a `class MyLaw : public Law`, declare its factory and entry below,
// add the entry to src/main.cpp, and list its source in CMakeLists.txt.
//
// Both laws fly fixed-wing aircraft only, design their gains in configure() for the
// telemetry's control period (std::runtime_error if the synthesis fails), and must be
// configured before their first step.

#ifndef ASTRUM_LAWS_HPP
#define ASTRUM_LAWS_HPP

#include <memory>

#include "astrum/runtime.hpp"

namespace astrum {

std::unique_ptr<Law> make_lqr_law(const LawContext& context);

// `horizon_s`: the MPC look-ahead, round(horizon_s / dt) steps of the control period.
// Throws std::invalid_argument unless it is finite and > 0; configure() throws
// std::invalid_argument if it covers less than one control period.
std::unique_ptr<Law> make_mpc_law(const LawContext& context, double horizon_s);

LawEntry lqr_law_entry();
// Registers --horizon-s (default 0.5 s).
LawEntry mpc_law_entry();

}  // namespace astrum

#endif  // ASTRUM_LAWS_HPP
