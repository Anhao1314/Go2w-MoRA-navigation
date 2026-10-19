#include "go2w/model.hpp"

namespace go2w {

RobotModel::RobotModel() {
  for (const auto& body : constants::kBodies) {
    total_mass_ += body.mass;
  }
}

Vec3 RobotModel::bodyInertialDiagonal(int index) const {
  const auto& d = constants::kBodies[index].diag_inertia;
  return Vec3(d[0], d[1], d[2]);
}

Eigen::Quaterniond RobotModel::bodyInertialQuaternion(int index) const {
  const auto& q = constants::kBodies[index].iquat;
  // Generated order is xyzw.
  return Eigen::Quaterniond(q[3], q[0], q[1], q[2]);
}

}  // namespace go2w
