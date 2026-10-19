#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <string>

#include "fixture_io.hpp"
#include "go2w/generated/model_constants.hpp"
#include "go2w/kinematics.hpp"
#include "go2w/model.hpp"
#include "go2w/path_tracker.hpp"

namespace {

std::string fixture(const char* name) {
  return std::string(GO2W_FIXTURE_DIR) + "/" + name;
}

}  // namespace

TEST_CASE("FK wheel centers match MuJoCo", "[kinematics][fixtures]") {
  const go2w::RobotModel model;
  const go2w::Kinematics kinematics(model);
  const auto rows = go2w::test::readNumericCsv(fixture("fk.csv"));
  REQUIRE(rows.size() == 100);

  double max_error = 0.0;
  for (const auto& row : rows) {
    const go2w::RobotState state = go2w::test::stateFromRow(row);
    for (int leg_i = 0; leg_i < go2w::kNumLegs; ++leg_i) {
      const go2w::Vec3 expected(row[23 + 3 * leg_i], row[24 + 3 * leg_i],
                                row[25 + 3 * leg_i]);
      const go2w::Vec3 actual =
          kinematics.wheelCenterWorld(state, static_cast<go2w::Leg>(leg_i));
      max_error = std::max(max_error, (actual - expected).cwiseAbs().maxCoeff());
    }
  }
  INFO("max FK error: " << max_error);
  REQUIRE(max_error < 1e-9);
}

TEST_CASE("Contact Jacobians match MuJoCo", "[kinematics][fixtures]") {
  const go2w::RobotModel model;
  const go2w::Kinematics kinematics(model);
  const auto rows = go2w::test::readNumericCsv(fixture("jac.csv"));
  REQUIRE(rows.size() == 100);

  double max_error = 0.0;
  for (const auto& row : rows) {
    const go2w::RobotState state = go2w::test::stateFromRow(row);
    for (int leg_i = 0; leg_i < go2w::kNumLegs; ++leg_i) {
      const int offset = 23 + leg_i * 3 * go2w::constants::kNumDof;
      Eigen::Matrix<double, 3, go2w::constants::kNumDof> expected;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < go2w::constants::kNumDof; ++c) {
          expected(r, c) = row[offset + r * go2w::constants::kNumDof + c];
        }
      }
      const auto J = kinematics.contactJacobianWorld(
          state, static_cast<go2w::Leg>(leg_i));
      max_error = std::max(max_error, (J - expected).cwiseAbs().maxCoeff());
    }
  }
  INFO("max Jacobian error: " << max_error);
  REQUIRE(max_error < 1e-6);
}

TEST_CASE("IK round trip reaches FK targets", "[kinematics]") {
  const go2w::RobotModel model;
  const go2w::Kinematics kinematics(model);
  const go2w::Vec3 seed(go2w::constants::kStandLeg[0],
                        go2w::constants::kStandLeg[1],
                        go2w::constants::kStandLeg[2]);
  const auto rows = go2w::test::readNumericCsv(fixture("fk.csv"));
  REQUIRE(rows.size() == 100);

  int solved = 0;
  double max_error = 0.0;
  for (int sample = 0; sample < 50; ++sample) {
    const auto& row = rows[sample];
    const go2w::RobotState state = go2w::test::stateFromRow(row);
    for (int leg_i = 0; leg_i < go2w::kNumLegs; ++leg_i) {
      const go2w::Vec3 q(row[7 + 3 * leg_i], row[8 + 3 * leg_i],
                          row[9 + 3 * leg_i]);
      const go2w::Vec3 target =
          kinematics.fkWheelBase(static_cast<go2w::Leg>(leg_i), q).translation();
      go2w::Vec3 q_out;
      if (kinematics.ikWheelBase(static_cast<go2w::Leg>(leg_i), target, seed,
                                 &q_out)) {
        ++solved;
        const go2w::Vec3 reached =
            kinematics.fkWheelBase(static_cast<go2w::Leg>(leg_i), q_out)
                .translation();
        max_error = std::max(max_error, (reached - target).norm());
      }
    }
  }
  INFO("IK solved: " << solved << " / 200, max target error: " << max_error);
  REQUIRE(solved >= 190);
  REQUIRE(max_error < 1e-5);
}

TEST_CASE("Scripted teacher parity with Python", "[path][fixtures]") {
  const go2w::PathTracker tracker(go2w::PathTrackerConfig{});
  const auto rows = go2w::test::readNumericCsv(fixture("parity.csv"));
  REQUIRE(rows.size() == 1920);

  double max_error = 0.0;
  for (const auto& row : rows) {
    const int segment_idx = static_cast<int>(row[5]);
    const go2w::Vec6 action = tracker.legacyAction(
        segment_idx, row[0], row[1], row[2], row[3], row[4], row[6], row[7]);
    for (int i = 0; i < 6; ++i) {
      max_error = std::max(max_error, std::abs(action[i] - row[8 + i]));
    }
  }
  INFO("max teacher parity error: " << max_error);
  // The Python controller returns float32 actions; 1e-6 covers float32
  // rounding while still catching real formula drift.
  REQUIRE(max_error < 1e-6);
}
