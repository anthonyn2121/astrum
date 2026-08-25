// MPC control law (skeleton).
//
// Receding-horizon (model-predictive) control. Each frame it predicts the
// error-state N steps ahead with a linear model, minimizes a quadratic cost over
// the input sequence, and applies the first move — then re-plans next frame.
//
// Like laws/lqr.cpp the channels are decoupled for clarity:
//
//   lateral      x_lat = [heading_err, roll,  roll_rate ]   -> aileron
//   longitudinal x_lon = [alt_err,     pitch, pitch_rate]   -> elevator
//   speed        x_spd = [airspeed_err]                     -> throttle
//
// Each channel is regulated to zero error. For the *unconstrained* quadratic
// problem the optimal first move is a constant linear gain on the current
// error-state (computed once in ChannelMpc), so the hot path is a single
// matrix-vector product. That gain approaches the LQR gain as N -> inf — the
// point of MPC is what you add next:
//
// Because heading and altitude respond over *seconds*, the model is discretized
// on a coarse step (--horizon-s / --horizon, ~0.1 s) so a modest step count still
// looks seconds ahead — the horizon must outlast the mode you're regulating, or
// the optimizer sees the control do nothing and won't use it. This model step is
// independent of the 120 Hz control rate: the loop re-solves and applies the
// first move on every telemetry frame regardless.
//
// What to do: (1) replace the placeholder per-channel (A, B) below with *your*
// linearized model, tune (Q, R) and the horizon; (2) add the hard limits LQR
// can't express — bank/pitch/actuator bounds — at the marked hook in
// ChannelMpc::move, which then needs a QP solver (e.g. OSQP) instead of the
// closed form. The shipped model flies `--law mpc` (its gains land near the
// proven LQR law) so you have a working loop to grow.

#include <Eigen/Dense>

#include "astrum_control.hpp"

namespace astrum {
namespace {

// Unconstrained condensed MPC for one linear channel x[k+1] = A x + B u.
// Regulates the error-state to zero over an N-step horizon with stage cost
// x'Qx + u'Ru. Precomputes the first-move feedback gain K so the optimal first
// input is u0 = -K x0.
class ChannelMpc {
 public:
  ChannelMpc(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
             const Eigen::MatrixXd& R, int N) {
    const int n = static_cast<int>(B.rows());
    const int m = static_cast<int>(B.cols());

    // Condensed prediction: X = Sx x0 + Su U, over k = 1..N.
    Eigen::MatrixXd Sx(N * n, n);
    Eigen::MatrixXd Su = Eigen::MatrixXd::Zero(N * n, N * m);
    Eigen::MatrixXd Apow = Eigen::MatrixXd::Identity(n, n);
    for (int i = 0; i < N; ++i) {
      Apow = Apow * A;  // A^(i+1)
      Sx.block(i * n, 0, n, n) = Apow;
      Eigen::MatrixXd AjB = B;  // A^0 B, then A^1 B, ...
      for (int j = i; j >= 0; --j) {
        // block (i, j) = A^(i-j) B ; iterate j downward so AjB tracks A^(i-j) B.
        Su.block(i * n, j * m, n, m) = AjB;
        AjB = A * AjB;
      }
    }

    // Block-diagonal weights (kron(I_N, Q/R)).
    Eigen::MatrixXd Qbar = Eigen::MatrixXd::Zero(N * n, N * n);
    Eigen::MatrixXd Rbar = Eigen::MatrixXd::Zero(N * m, N * m);
    for (int i = 0; i < N; ++i) {
      Qbar.block(i * n, i * n, n, n) = Q;
      Rbar.block(i * m, i * m, m, m) = R;
    }

    // U* = -(Su'Qbar Su + Rbar)^-1 Su'Qbar Sx x0 ; keep the first m rows.
    const Eigen::MatrixXd H = Su.transpose() * Qbar * Su + Rbar;
    const Eigen::MatrixXd Kfull = H.ldlt().solve(Su.transpose() * Qbar * Sx);
    k_ = Kfull.topRows(m);
  }

  // First optimal input for the current error-state x0.
  Eigen::VectorXd move(const Eigen::VectorXd& x0) const {
    // ---- CONSTRAINT HOOK --------------------------------------------------
    // Unconstrained closed form. To enforce hard limits (|u| <= umax, bank <=
    // phi_max, ...), replace this with a QP over the horizon subject to those
    // bounds (OSQP) and return the first solved input.
    return -k_ * x0;
  }

  double scalar_move(const Eigen::VectorXd& x0) const { return move(x0)(0); }

 private:
  Eigen::MatrixXd k_;
};

// Euler-discretized placeholder model for one 2nd-order attitude channel
// [track_err, angle, angle_rate] with a single actuator. `couple` is the
// track-error/angle coupling (e.g. -g/V for heading-vs-bank, -V for altitude-vs-
// pitch); `b` the actuator->angular-accel gain (sign encodes actuator polarity).
Eigen::MatrixXd attitude_A(double couple, double dt) {
  Eigen::Matrix3d A;
  A << 1.0, couple * dt, 0.0,
       0.0, 1.0, dt,
       0.0, 0.0, 1.0 - 2.0 * dt;  // 2.0 = placeholder angular-rate damping
  return A;
}

class MpcLaw : public Law {
 public:
  explicit MpcLaw(const Cli& cli) {
    const int N = cli.horizon;
    const double dt = cli.horizon_s / N;  // coarse model step (~0.1s), NOT the control rate

    // --- PLACEHOLDER MODELS (Euler-discretized). Replace with your own. ---
    // Continuous-time constants (all PLACEHOLDER — swap in your linearization):
    const double g_over_v = 9.81 / 50.0;  // turn-rate per unit bank at ~50 m/s
    const double v = 50.0;                // airspeed for the climb-rate coupling
    const double b = 4.0;                 // actuator -> angular acceleration

    // Lateral: heading changes with bank; roll is a damped 2nd-order response to
    // aileron. State [heading_err, roll, roll_rate], input aileron.
    Eigen::Vector3d B_lat(0.0, 0.0, b * dt);
    Eigen::Vector3d q_lat(6.0, 1.0, 0.1);
    lat_ = std::make_unique<ChannelMpc>(attitude_A(-g_over_v, dt), B_lat,
                                        q_lat.asDiagonal(), r1(1.0), N);

    // Longitudinal: altitude changes with pitch; pitch is a damped response to
    // elevator (elevator+ pitches DOWN, hence -B). State [alt_err, pitch,
    // pitch_rate], input elevator.
    Eigen::Vector3d B_lon(0.0, 0.0, -b * dt);
    Eigen::Vector3d q_lon(0.02, 1.0, 0.05);
    lon_ = std::make_unique<ChannelMpc>(attitude_A(-v, dt), B_lon,
                                        q_lon.asDiagonal(), r1(16.0), N);

    // Speed: single integrator; more throttle raises airspeed (reduces error).
    Eigen::MatrixXd A_spd(1, 1);
    A_spd << 1.0;
    Eigen::MatrixXd B_spd(1, 1);
    B_spd << -b * dt;
    spd_ = std::make_unique<ChannelMpc>(A_spd, B_spd, r1(1.0), r1(500.0), N);
  }

  Command control(const VehicleState& s, const Reference& ref) override {
    const double heading_err = wrap_pi(ref.heading_rad - s.yaw);
    const double alt_err = ref.alt_m - s.alt_m;
    const double airspeed_err = ref.airspeed_mps - s.airspeed;

    Eigen::Vector3d x_lat(heading_err, s.roll, s.p);
    Eigen::Vector3d x_lon(alt_err, s.pitch, s.q);
    Eigen::VectorXd x_spd(1);
    x_spd << airspeed_err;

    const double aileron = lat_->scalar_move(x_lat);
    const double elevator = lon_->scalar_move(x_lon);
    const double throttle = kThrottleTrim + spd_->scalar_move(x_spd);
    const double rudder = 0.0;  // coordinated-turn assumption; not modeled here

    return Command::raw(aileron, elevator, rudder, throttle);
  }

 private:
  // 1x1 weight matrix helper (keeps the channel construction readable).
  static Eigen::MatrixXd r1(double v) {
    Eigen::MatrixXd m(1, 1);
    m << v;
    return m;
  }

  static constexpr double kThrottleTrim = 0.6;
  std::unique_ptr<ChannelMpc> lat_, lon_, spd_;
};

}  // namespace

std::unique_ptr<Law> make_mpc_law(const Cli& cli) { return std::make_unique<MpcLaw>(cli); }

}  // namespace astrum
