#pragma once

#include "go2w/kinematics.hpp"
#include "go2w/model.hpp"
#include "go2w/types.hpp"

namespace go2w {

class Dynamics {
 public:
  explicit Dynamics(const RobotModel& model);

  CentroidalState computeCentroidal(const RobotState& state) const;

 private:
  const RobotModel& model_;
  Kinematics kinematics_;
};

}  // namespace go2w
