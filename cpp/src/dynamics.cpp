#include "go2w/dynamics.hpp"

#include <array>

namespace go2w {
namespace {

struct JointFrame {
  Vec3 axis_world = Vec3::Zero();
  Vec3 anchor_world = Vec3::Zero();
  double qd = 0.0;
};

struct LegFrames {
  std::array<JointFrame, 4> joints;
};

std::array<LegFrames, kNumLegs> computeLegFrames(const RobotModel& model,
                                                 const RobotState& state,
                                                 const BodyTransforms& transforms) {
  std::array<LegFrames, kNumLegs> out;
  for (int leg_i = 0; leg_i < kNumLegs; ++leg_i) {
    const auto& leg_def = model.leg(leg_i);
    Eigen::Isometry3d T = transforms.T[0];
    for (int k = 0; k < 4; ++k) {
      const auto& body = model.body(leg_def.body[k]);
      const Vec3 pos(body.pos[0], body.pos[1], body.pos[2]);
      const Vec3 axis(body.axis[0], body.axis[1], body.axis[2]);
      T = T * Eigen::Translation3d(pos);
      out[leg_i].joints[k].axis_world = T.rotation() * axis.normalized();
      out[leg_i].joints[k].anchor_world = T.translation();
      out[leg_i].joints[k].qd =
          (k == 3) ? state.joints.qd_wheel[leg_i]
                   : state.joints.qd_leg[3 * leg_i + k];
      const double q = (k == 3) ? state.joints.q_wheel[leg_i]
                                : state.joints.q_leg[3 * leg_i + k];
      T = T * Eigen::AngleAxisd(q, axis.normalized());
    }
  }
  return out;
}

}  // namespace

Dynamics::Dynamics(const RobotModel& model) : model_(model), kinematics_(model) {}

CentroidalState Dynamics::computeCentroidal(const RobotState& state) const {
  const BodyTransforms transforms = kinematics_.computeBodyTransforms(state);
  const auto leg_frames = computeLegFrames(model_, state, transforms);

  CentroidalState out;
  out.mass = model_.totalMass();
  Vec3 weighted_com = Vec3::Zero();
  Vec3 weighted_vel = Vec3::Zero();

  std::array<Vec3, constants::kNumBodies> body_inertial_positions;
  std::array<Vec3, constants::kNumBodies> body_velocities;

  for (int i = 0; i < constants::kNumBodies; ++i) {
    const auto& body = model_.body(i);
    const Vec3 ipos(body.ipos[0], body.ipos[1], body.ipos[2]);
    const Vec3 p_i = transforms.T[i] * ipos;
    body_inertial_positions[i] = p_i;

    Vec3 v_i = state.base.v + state.base.w.cross(p_i - state.base.p);
    if (body.leg >= 0 && body.joint_kind >= 0) {
      const int leg_i = body.leg;
      const int last = body.joint_kind;
      for (int k = 0; k <= last; ++k) {
        const auto& jf = leg_frames[leg_i].joints[k];
        v_i += jf.axis_world.cross(p_i - jf.anchor_world) * jf.qd;
      }
    }
    body_velocities[i] = v_i;
    weighted_com += body.mass * p_i;
    weighted_vel += body.mass * v_i;
  }

  out.com = weighted_com / out.mass;
  out.com_vel = weighted_vel / out.mass;

  Mat3 inertia = Mat3::Zero();
  for (int i = 0; i < constants::kNumBodies; ++i) {
    const auto& body = model_.body(i);
    const Eigen::Matrix3d R_body = transforms.T[i].rotation();
    const Eigen::Matrix3d R_inertial =
        R_body * model_.bodyInertialQuaternion(i).toRotationMatrix();
    const Vec3 diag = model_.bodyInertialDiagonal(i);
    const Mat3 I_world =
        R_inertial * diag.asDiagonal() * R_inertial.transpose();
    const Vec3 r = body_inertial_positions[i] - out.com;
    inertia += I_world +
               body.mass * (r.squaredNorm() * Mat3::Identity() - r * r.transpose());
  }
  out.inertia_com = inertia;
  return out;
}

}  // namespace go2w
