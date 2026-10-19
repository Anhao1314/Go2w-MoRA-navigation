#include <iostream>

#include "go2w/controller.hpp"

int main() {
  go2w::S0Controller controller;
  go2w::RobotState state;
  state.base.p = Eigen::Vector3d(1.0, 0.0, go2w::constants::kStandBaseHeight);
  state.base.q = Eigen::Quaterniond::Identity();
  state.joints.q_leg.setZero();
  const go2w::ControlOutput out =
      controller.stepLegacy(state, 1.0, 0.0, 0, 1.0, 0.0);
  std::cout << "action6 = " << out.action6.transpose() << "\n";
  std::cout << "total mass = " << controller.model().totalMass() << " kg\n";
  std::cout << "safety = " << static_cast<int>(out.safety) << "\n";
  return 0;
}
