#pragma once

#include <array>

#include "go2w/types.hpp"

namespace go2w {

struct MpcConfig {
  int horizon = 20;
  double dt = 0.02;  // 50 Hz
  double mu = 0.6;   // conservative vs the model's 0.8
  double fz_max = 150.0;
  double f_max = 200.0;
  double slack_penalty = 1.0e4;
  double z_min = 0.20;
  double z_max = 0.80;
  double roll_pitch_max = 0.35;
  std::array<double, 12> q{
      1.0, 1.0, 10.0,   // p
      10.0, 10.0, 10.0, // v
      100.0, 100.0, 50.0, // theta
      10.0, 10.0, 10.0, // omega
  };
  std::array<double, 12> q_terminal{
      10.0, 10.0, 20.0,
      20.0, 20.0, 20.0,
      200.0, 200.0, 100.0,
      20.0, 20.0, 20.0,
  };
  std::array<double, 12> r{
      1.0e-3, 1.0e-3, 1.0e-3,
      1.0e-3, 1.0e-3, 1.0e-3,
      1.0e-3, 1.0e-3, 1.0e-3,
      1.0e-3, 1.0e-3, 1.0e-3,
  };
};

struct MpcInput {
  double mass = 19.126408;
  Vec3 com = Vec3::Zero();
  Vec3 com_vel = Vec3::Zero();
  Mat3 inertia_com = Mat3::Identity();
  Vec3 theta = Vec3::Zero();  // roll, pitch, yaw
  Vec3 omega = Vec3::Zero();
  std::array<Vec3, kNumLegs> contact_points_world{{
      Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()}};
  Vec3 v_ref = Vec3::Zero();
  double yaw_ref = 0.0;
  double wz_ref = 0.0;
  double z_ref = 0.436;
};

struct MpcOutput {
  std::array<Vec3, kNumLegs> forces{{
      Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero()}};
  bool success = false;
  int status = 0;
  double solve_ms = 0.0;
  double cost = 0.0;
  int iterations = 0;
};

class SrbdMpc {
 public:
  explicit SrbdMpc(const MpcConfig& config = {});
  MpcOutput solve(const MpcInput& input);
  const MpcConfig& config() const { return config_; }

 private:
  MpcConfig config_;
};

}  // namespace go2w
