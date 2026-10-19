#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <cmath>

#include "go2w/dynamics.hpp"
#include "go2w/kinematics.hpp"
#include "go2w/model.hpp"
#include "go2w/mpc.hpp"

namespace {

go2w::MpcInput makeStandInput() {
  const go2w::RobotModel model;
  const go2w::Kinematics kinematics(model);
  const go2w::Dynamics dynamics(model);

  go2w::RobotState state;
  state.base.p = go2w::Vec3(0.0, 0.0, go2w::constants::kStandBaseHeight);
  state.base.q = Eigen::Quaterniond::Identity();
  for (int leg = 0; leg < go2w::kNumLegs; ++leg) {
    state.joints.q_leg.segment<3>(3 * leg) =
        go2w::Vec3(go2w::constants::kStandLeg[0],
                   go2w::constants::kStandLeg[1],
                   go2w::constants::kStandLeg[2]);
  }

  const auto transforms = kinematics.computeBodyTransforms(state);
  go2w::MpcInput input;
  const auto centroidal = dynamics.computeCentroidal(state);
  input.mass = centroidal.mass;
  input.com = centroidal.com;
  input.com_vel = go2w::Vec3::Zero();
  input.inertia_com = centroidal.inertia_com;
  input.theta = go2w::Vec3::Zero();
  input.omega = go2w::Vec3::Zero();
  input.v_ref = go2w::Vec3::Zero();
  input.yaw_ref = 0.0;
  input.wz_ref = 0.0;
  input.z_ref = centroidal.com.z();
  for (int leg = 0; leg < go2w::kNumLegs; ++leg) {
    const int body = model.wheelBodyIndex(static_cast<go2w::Leg>(leg));
    input.contact_points_world[leg] =
        transforms.T[body].translation() -
        go2w::Vec3(0.0, 0.0, go2w::constants::kWheelRadius);
  }
  return input;
}

double sumZ(const go2w::MpcOutput& output) {
  double total = 0.0;
  for (const auto& force : output.forces) {
    total += force.z();
  }
  return total;
}

double sumX(const go2w::MpcOutput& output) {
  double total = 0.0;
  for (const auto& force : output.forces) {
    total += force.x();
  }
  return total;
}

}  // namespace

TEST_CASE("SRBD MPC hover solution supports weight", "[mpc]") {
  const go2w::MpcInput input = makeStandInput();
  go2w::SrbdMpc mpc;
  const go2w::MpcOutput output = mpc.solve(input);

  INFO("status: " << output.status << " iterations: " << output.iterations
                  << " solve_ms: " << output.solve_ms);
  REQUIRE(output.success);

  const double weight = input.mass * 9.81;
  REQUIRE(std::abs(sumZ(output) - weight) < 0.20 * weight);
  const double friction_limit = mpc.config().mu / std::sqrt(2.0);
  for (const auto& force : output.forces) {
    REQUIRE(force.z() >= -1e-6);
    REQUIRE(std::abs(force.x()) <= friction_limit * force.z() + 1e-3);
    REQUIRE(std::abs(force.y()) <= friction_limit * force.z() + 1e-3);
  }
  REQUIRE(output.solve_ms < 100.0);
}

TEST_CASE("SRBD MPC produces forward force for forward velocity reference",
          "[mpc]") {
  go2w::MpcInput input = makeStandInput();
  input.v_ref = go2w::Vec3(0.5, 0.0, 0.0);
  go2w::SrbdMpc mpc;
  const go2w::MpcOutput output = mpc.solve(input);
  INFO("status: " << output.status << " solve_ms: " << output.solve_ms);
  REQUIRE(output.success);
  REQUIRE(sumX(output) > 0.0);
}

TEST_CASE("SRBD MPC solve time benchmark", "[mpc][benchmark]") {
  const go2w::MpcInput input = makeStandInput();
  go2w::SrbdMpc mpc;
  double total_ms = 0.0;
  int solved = 0;
  for (int i = 0; i < 20; ++i) {
    const go2w::MpcOutput output = mpc.solve(input);
    if (output.success) {
      ++solved;
    }
    total_ms += output.solve_ms;
  }
  INFO("solved " << solved << "/20, average solve_ms: " << total_ms / 20.0);
  REQUIRE(solved >= 18);
  REQUIRE(total_ms / 20.0 < 50.0);
}
