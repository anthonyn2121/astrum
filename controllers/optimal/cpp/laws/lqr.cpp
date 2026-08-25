// LQR control law (skeleton).
//
// Full-state feedback u = -K (x - x_ref) on a linearized model of the aircraft.
// This file is where *your* control-theory work goes; the runtime
// (astrum_control) hands you a decoded VehicleState each frame and publishes
// whatever Command you return.
//
// The channels are kept decoupled for clarity (each is a small SISO loop):
//
//   lateral      x_lat = [heading_err, roll,  roll_rate ]   -> aileron
//   longitudinal x_lon = [alt_err,     pitch, pitch_rate]   -> elevator
//   speed        x_spd = [airspeed_err]                     -> throttle
//
// What to do: replace the hand-tuned placeholder gains below with the LQR
// solution of your own model. Linearize the aircraft about trim to get each
// channel's (A, B), choose weights (Q, R), and call astrum::dlqr (optimal_math.hpp)
// to get K. The placeholder gains reproduce the proven example controller's hold
// law so `--law lqr` flies out of the box while you develop the real thing.
//
// Note LQR is *unconstrained*: it can't respect a bank-angle or actuator limit
// except by weighting. When you need hard limits (max bank, elevator
// saturation), that's the motivation for `--law mpc`.

#include <Eigen/Dense>

#include "astrum_control.hpp"
#include "optimal_math.hpp"  // astrum::dlqr — for when you swap in a real model

namespace astrum {
namespace {

class LqrLaw : public Law {
 public:
  Command control(const VehicleState& s, const Reference& ref) override {
    // Errors relative to the reference (heading wrapped to [-pi, pi]).
    const double heading_err = wrap_pi(ref.heading_rad - s.yaw);
    const double alt_err = ref.alt_m - s.alt_m;
    const double airspeed_err = ref.airspeed_mps - s.airspeed;

    // State vectors, then u = -K x per channel. Signs of K already fold in the
    // error convention above (positive heading_err -> roll right, etc.).
    const Eigen::Vector3d x_lat(heading_err, s.roll, s.p);
    const Eigen::Vector3d x_lon(alt_err, s.pitch, s.q);

    const double aileron = -k_lat_.dot(x_lat);
    const double elevator = -k_lon_.dot(x_lon);
    const double throttle = kThrottleTrim - k_spd_ * airspeed_err;
    const double rudder = 0.0;  // coordinated-turn assumption; not modeled here

    return Command::raw(aileron, elevator, rudder, throttle);
  }

 private:
  // --- PLACEHOLDER GAINS ---------------------------------------------------
  // Hand-tuned to match the example controller's proven hold law so the loop
  // flies today. Replace each with astrum::dlqr(A, B, Q, R) for your linearized
  // model — e.g. for the lateral channel:
  //
  //     const double dt = 1.0 / 120.0;                 // sim rate
  //     Eigen::Matrix3d A_lat; A_lat << 1, -V*dt, 0,    // heading_err, roll, roll_rate
  //                                     0,  1,    dt,
  //                                     0,  0,    1 - a*dt;
  //     Eigen::Vector3d B_lat(0, 0, b*dt);             // aileron -> roll accel
  //     Eigen::RowVector3d K_lat = dlqr(A_lat, B_lat, Q_lat, R_ail);
  //
  // u = -K x, with x measured relative to the reference.
  const Eigen::RowVector3d k_lat_{-1.8, 1.5, 0.3};    // [hdg_err, roll, roll_rate] -> aileron
  const Eigen::RowVector3d k_lon_{0.02, -2.0, -0.5};  // [alt_err, pitch, pitch_rate] -> elevator
  const double k_spd_ = -0.02;                        // [airspeed_err] -> (throttle - trim)
  static constexpr double kThrottleTrim = 0.6;
};

}  // namespace

std::unique_ptr<Law> make_lqr_law(const Cli&) { return std::make_unique<LqrLaw>(); }

}  // namespace astrum
