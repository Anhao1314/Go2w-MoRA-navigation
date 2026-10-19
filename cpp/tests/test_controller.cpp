#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>

#include "go2w/controller.hpp"
#include "go2w/force_mapper.hpp"
#include "go2w/safety.hpp"

TEST_CASE("Scripted controller returns bounded actions", "[controller]") {
  go2w::S0Controller controller;
  go2w::RobotState state;
  state.base.p = go2w::Vec3(1.0, 0.0, go2w::constants::kStandBaseHeight);
  state.base.q = Eigen::Quaterniond::Identity();
  const auto output = controller.stepLegacy(state, 1.2, 0.1, 0, 1.0, 0.5);
  REQUIRE(output.valid);
  REQUIRE(output.action6.cwiseAbs().maxCoeff() <= 1.0);
  REQUIRE(output.safety == go2w::SafetyState::TRACK);
}

TEST_CASE("Force mapper produces zero longitudinal torque for vertical force",
          "[force]") {
  go2w::ForceMapper mapper(go2w::ForceMapperConfig{});
  std::array<go2w::Vec3, go2w::kNumLegs> forces;
  const double weight = 19.126408 * 9.81;
  for (auto& force : forces) {
    force = go2w::Vec3(0.0, 0.0, weight / 4.0);
  }
  const auto torques = mapper.wheelTorques(forces, Eigen::Quaterniond::Identity());
  for (double torque : torques) {
    REQUIRE(std::abs(torque) < 1e-12);
  }
}

TEST_CASE("Safety layer enters ESTOP at low height", "[safety]") {
  go2w::SafetyLayer safety;
  go2w::RobotState state;
  state.base.p = go2w::Vec3(0.0, 0.0, 0.05);
  go2w::HighLevelCommand command;
  const auto safety_state = safety.update(state, command, 0.01, true);
  REQUIRE(safety_state == go2w::SafetyState::ESTOP);
  REQUIRE(safety.filterAction(go2w::Vec6::Ones(), safety_state).isZero());
}

TEST_CASE("Safety layer degrades after repeated QP failure", "[safety]") {
  go2w::SafetyLayer safety;
  go2w::RobotState state;
  state.base.p = go2w::Vec3(0.0, 0.0, 0.436);
  state.base.q = Eigen::Quaterniond::Identity();
  go2w::HighLevelCommand command;
  go2w::SafetyState result = go2w::SafetyState::INIT;
  for (int i = 0; i < 20; ++i) {
    result = safety.update(state, command, 0.01, false);
  }
  REQUIRE(result == go2w::SafetyState::DEGRADED);
}
