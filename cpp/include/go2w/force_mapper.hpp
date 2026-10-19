#pragma once

#include <array>

#include "go2w/generated/model_constants.hpp"
#include "go2w/types.hpp"

namespace go2w {

struct ForceMapperConfig {
  double wheel_radius = constants::kWheelRadius;
  double wheel_torque_limit = constants::kTorqueLimitWheel;
  double torque_sign = 1.0;
};

class ForceMapper {
 public:
  explicit ForceMapper(const ForceMapperConfig& config);

  // Contact forces are the forces the ground exerts on the robot, expressed in
  // the world frame. The returned vector follows the env's 6D action layout:
  // [FR, FL, RR, RL] normalized wheel torques, then two leg-group commands.
  Vec6 toLegacyAction(const std::array<Vec3, kNumLegs>& force_world,
                      const Eigen::Quaterniond& base_q) const;

  // Wheel torques in Nm, FR/FL/RR/RL order.
  std::array<double, kNumLegs> wheelTorques(
      const std::array<Vec3, kNumLegs>& force_world,
      const Eigen::Quaterniond& base_q) const;

  const ForceMapperConfig& config() const { return config_; }

 private:
  ForceMapperConfig config_;
};

}  // namespace go2w
