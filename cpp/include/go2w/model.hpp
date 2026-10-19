#pragma once

#include "go2w/generated/model_constants.hpp"
#include "go2w/types.hpp"

namespace go2w {

class RobotModel {
 public:
  RobotModel();

  int numBodies() const { return constants::kNumBodies; }
  const constants::BodyDef& body(int index) const { return constants::kBodies[index]; }
  const constants::LegDef& leg(int index) const { return constants::kLegs[index]; }
  const constants::LegDef& leg(Leg leg) const {
    return constants::kLegs[static_cast<int>(leg)];
  }

  double totalMass() const { return total_mass_; }
  Vec3 bodyInertialDiagonal(int index) const;
  Eigen::Quaterniond bodyInertialQuaternion(int index) const;
  int wheelBodyIndex(Leg leg) const { return constants::kLegs[static_cast<int>(leg)].body[3]; }

 private:
  double total_mass_ = 0.0;
};

}  // namespace go2w
