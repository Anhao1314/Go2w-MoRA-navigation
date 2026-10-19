#include "go2w/force_mapper.hpp"

#include <algorithm>

namespace go2w {

ForceMapper::ForceMapper(const ForceMapperConfig& config) : config_(config) {}

std::array<double, kNumLegs> ForceMapper::wheelTorques(
    const std::array<Vec3, kNumLegs>& force_world,
    const Eigen::Quaterniond& base_q) const {
  const Eigen::Matrix3d R = base_q.toRotationMatrix();
  std::array<double, kNumLegs> torques{};
  for (int i = 0; i < kNumLegs; ++i) {
    const Vec3 force_base = R.transpose() * force_world[i];
    const double tau = config_.torque_sign * config_.wheel_radius * force_base.x();
    torques[i] = std::max(-config_.wheel_torque_limit,
                          std::min(config_.wheel_torque_limit, tau));
  }
  return torques;
}

Vec6 ForceMapper::toLegacyAction(const std::array<Vec3, kNumLegs>& force_world,
                                 const Eigen::Quaterniond& base_q) const {
  const auto torques = wheelTorques(force_world, base_q);
  Vec6 action;
  for (int i = 0; i < kNumLegs; ++i) {
    action[i] = torques[i] / config_.wheel_torque_limit;
  }
  action[4] = 0.0;
  action[5] = 0.0;
  return action.cwiseMax(-1.0).cwiseMin(1.0);
}

}  // namespace go2w
