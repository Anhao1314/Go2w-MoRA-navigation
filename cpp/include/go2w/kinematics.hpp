#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>

#include "go2w/generated/model_constants.hpp"
#include "go2w/model.hpp"
#include "go2w/types.hpp"

namespace go2w {

struct BodyTransforms {
  std::array<Eigen::Isometry3d, constants::kNumBodies> T;
};

class Kinematics {
 public:
  explicit Kinematics(const RobotModel& model);

  BodyTransforms computeBodyTransforms(const RobotState& state) const;

  Eigen::Isometry3d fkWheelBase(Leg leg, const Vec3& q_leg) const;
  Vec3 wheelCenterWorld(const RobotState& state, Leg leg) const;
  Mat3 wheelJacobianBase(Leg leg, const Vec3& q_leg) const;
  // Jacobian of one wheel center in the world frame. The free-base block is
  // point-dependent, so it is computed per leg.
  Eigen::Matrix<double, 3, constants::kNumDof> contactJacobianWorld(
      const RobotState& state, Leg leg) const;
  // Stacked 12 x nv Jacobian for all four wheel centers, in FR/FL/RR/RL order.
  Eigen::Matrix<double, 12, constants::kNumDof> contactJacobianWorldAll(
      const RobotState& state) const;
  Vec3 contactPointWorld(const RobotState& state, Leg leg,
                         const Vec3& ground_normal) const;
  bool ikWheelBase(Leg leg, const Vec3& p_target_base, const Vec3& q_seed,
                   Vec3* q_out, int max_iterations = 60,
                   double tolerance = 1e-8) const;

 private:
  const RobotModel& model_;
};

}  // namespace go2w
