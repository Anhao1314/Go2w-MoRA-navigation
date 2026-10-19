#include "go2w/kinematics.hpp"

#include <algorithm>
#include <cmath>

namespace go2w {
namespace {

double jointAngleForBody(const RobotState& state, const constants::BodyDef& body) {
  if (body.leg < 0 || body.joint_kind < 0) {
    return 0.0;
  }
  if (body.joint_kind == 3) {
    return state.joints.q_wheel[body.leg];
  }
  return state.joints.q_leg[3 * body.leg + body.joint_kind];
}

Eigen::Matrix3d skew(const Vec3& v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

}  // namespace

Kinematics::Kinematics(const RobotModel& model) : model_(model) {}

BodyTransforms Kinematics::computeBodyTransforms(const RobotState& state) const {
  BodyTransforms out;
  for (int i = 0; i < constants::kNumBodies; ++i) {
    const auto& body = constants::kBodies[i];
    if (i == 0) {
      out.T[i] = Eigen::Isometry3d::Identity();
      out.T[i].linear() = state.base.q.toRotationMatrix();
      out.T[i].translation() = state.base.p;
      continue;
    }
    const int parent = body.parent;
    if (parent < 0 || parent >= i) {
      out.T[i] = Eigen::Isometry3d::Identity();
      continue;
    }
    const Vec3 pos(body.pos[0], body.pos[1], body.pos[2]);
    const Vec3 axis(body.axis[0], body.axis[1], body.axis[2]);
    const double q = jointAngleForBody(state, body);
    Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
    local.translation() = pos;
    if (axis.squaredNorm() > 1e-12) {
      local.linear() = Eigen::AngleAxisd(q, axis.normalized()).toRotationMatrix();
    }
    out.T[i] = out.T[parent] * local;
  }
  return out;
}

Eigen::Isometry3d Kinematics::fkWheelBase(Leg leg, const Vec3& q_leg) const {
  RobotState state;
  state.base.p.setZero();
  state.base.q.setIdentity();
  const int leg_idx = legIndex(leg);
  state.joints.q_leg.segment<3>(3 * leg_idx) = q_leg;
  const BodyTransforms transforms = computeBodyTransforms(state);
  return transforms.T[model_.wheelBodyIndex(leg)];
}

Vec3 Kinematics::wheelCenterWorld(const RobotState& state, Leg leg) const {
  const BodyTransforms transforms = computeBodyTransforms(state);
  return transforms.T[model_.wheelBodyIndex(leg)].translation();
}

Mat3 Kinematics::wheelJacobianBase(Leg leg, const Vec3& q_leg) const {
  const auto& leg_def = model_.leg(leg);
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  std::array<Vec3, 3> axes;
  std::array<Vec3, 3> anchors;

  for (int k = 0; k < 3; ++k) {
    const auto& body = constants::kBodies[leg_def.body[k]];
    const Vec3 pos(body.pos[0], body.pos[1], body.pos[2]);
    const Vec3 axis(body.axis[0], body.axis[1], body.axis[2]);
    T = T * Eigen::Translation3d(pos);
    axes[k] = T.rotation() * axis.normalized();
    anchors[k] = T.translation();
    T = T * Eigen::AngleAxisd(q_leg[k], axis.normalized());
  }

  const auto& wheel_body = constants::kBodies[leg_def.body[3]];
  const Vec3 wheel_pos(wheel_body.pos[0], wheel_body.pos[1], wheel_body.pos[2]);
  const Vec3 p = (T * Eigen::Translation3d(wheel_pos)).translation();

  Mat3 J;
  for (int k = 0; k < 3; ++k) {
    J.col(k) = axes[k].cross(p - anchors[k]);
  }
  return J;
}

Eigen::Matrix<double, 3, constants::kNumDof> Kinematics::contactJacobianWorld(
    const RobotState& state, Leg leg) const {
  Eigen::Matrix<double, 3, constants::kNumDof> J =
      Eigen::Matrix<double, 3, constants::kNumDof>::Zero();
  const BodyTransforms transforms = computeBodyTransforms(state);
  const Eigen::Matrix3d R_base = state.base.q.toRotationMatrix();
  const int leg_i = legIndex(leg);
  const auto& leg_def = constants::kLegs[leg_i];
  const Vec3 p = transforms.T[model_.wheelBodyIndex(leg)].translation();

  // Free-joint columns. MuJoCo's free-joint angular velocity is expressed in
  // the body frame, so rotate it with R_base. Verified by the fixtures.
  J.block<3, 3>(0, constants::kFreeDof) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(0, constants::kFreeDof + 3) =
      -skew(p - state.base.p) * R_base;

  Eigen::Isometry3d T = transforms.T[0];
  for (int k = 0; k < 3; ++k) {
    const auto& body = constants::kBodies[leg_def.body[k]];
    const Vec3 pos(body.pos[0], body.pos[1], body.pos[2]);
    const Vec3 axis(body.axis[0], body.axis[1], body.axis[2]);
    T = T * Eigen::Translation3d(pos);
    const Vec3 axis_world = T.rotation() * axis.normalized();
    const Vec3 anchor = T.translation();
    T = T * Eigen::AngleAxisd(state.joints.q_leg[3 * leg_i + k],
                              axis.normalized());
    J.col(leg_def.dof[k]) = axis_world.cross(p - anchor);
  }
  // The wheel joint axis passes through the wheel center, so its column is
  // zero for the contact point.
  J.col(leg_def.dof[3]).setZero();
  return J;
}

Eigen::Matrix<double, 12, constants::kNumDof>
Kinematics::contactJacobianWorldAll(const RobotState& state) const {
  Eigen::Matrix<double, 12, constants::kNumDof> J;
  for (int leg_i = 0; leg_i < kNumLegs; ++leg_i) {
    J.template block<3, constants::kNumDof>(3 * leg_i, 0) =
        contactJacobianWorld(state, static_cast<Leg>(leg_i));
  }
  return J;
}

Vec3 Kinematics::contactPointWorld(const RobotState& state, Leg leg,
                                   const Vec3& ground_normal) const {
  return wheelCenterWorld(state, leg) - constants::kWheelRadius * ground_normal;
}

bool Kinematics::ikWheelBase(Leg leg, const Vec3& p_target_base,
                             const Vec3& q_seed, Vec3* q_out,
                             int max_iterations, double tolerance) const {
  if (q_out == nullptr) {
    return false;
  }
  const auto& leg_def = model_.leg(leg);
  Vec3 q = q_seed;
  for (int i = 0; i < 3; ++i) {
    const double lo = leg_def.range[i][0];
    const double hi = leg_def.range[i][1];
    q[i] = std::max(lo, std::min(hi, q[i]));
  }

  for (int iter = 0; iter < max_iterations; ++iter) {
    const Vec3 p = fkWheelBase(leg, q).translation();
    const Vec3 error = p_target_base - p;
    if (error.norm() < tolerance) {
      *q_out = q;
      return true;
    }
    const Mat3 J = wheelJacobianBase(leg, q);
    const Mat3 JtJ = J.transpose() * J;
    const Mat3 damped = JtJ + 1e-6 * Eigen::Matrix3d::Identity();
    const Vec3 dq = damped.ldlt().solve(J.transpose() * error);
    for (int i = 0; i < 3; ++i) {
      const double lo = leg_def.range[i][0];
      const double hi = leg_def.range[i][1];
      q[i] = std::max(lo, std::min(hi, q[i] + dq[i]));
    }
    if (dq.norm() < 1e-12) {
      break;
    }
  }

  const Vec3 final_error = p_target_base - fkWheelBase(leg, q).translation();
  *q_out = q;
  return final_error.norm() < 1e-5;
}

}  // namespace go2w
