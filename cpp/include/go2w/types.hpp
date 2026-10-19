#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>

namespace go2w {

enum class Leg : int { FR = 0, FL = 1, RR = 2, RL = 3 };
constexpr int kNumLegs = 4;
constexpr int kNumLegJoints = 12;
constexpr int kNumWheels = 4;
constexpr int kNumActuators = 16;

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Vec16 = Eigen::Matrix<double, 16, 1>;
using Vec22 = Eigen::Matrix<double, 22, 1>;
using Mat3 = Eigen::Matrix3d;

inline int legIndex(Leg leg) { return static_cast<int>(leg); }

struct BaseState {
  Vec3 p = Vec3::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Vec3 v = Vec3::Zero();  // world-frame linear velocity
  Vec3 w = Vec3::Zero();  // world-frame angular velocity
};

struct JointState {
  Vec12 q_leg = Vec12::Zero();
  Vec12 qd_leg = Vec12::Zero();
  Vec4 q_wheel = Vec4::Zero();
  Vec4 qd_wheel = Vec4::Zero();
  Vec16 tau = Vec16::Zero();
};

struct RobotState {
  double t = 0.0;
  BaseState base;
  JointState joints;
};

enum class SafetyState {
  INIT = 0,
  STAND = 1,
  TRACK = 2,
  DEGRADED = 3,
  SAFE_STOP = 4,
  ESTOP = 5,
};

struct HighLevelCommand {
  enum class Mode { LEGACY_6D = 0, PHYSICAL_VELOCITY = 1, SAFE_STOP = 2 };
  Mode mode = Mode::LEGACY_6D;
  double speed_scale = 1.0;
  double turn_adjust = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double wz = 0.0;
  int segment_idx = 0;
  double dist_to_subgoal = 0.0;
  double speed = 0.0;
};

struct ContactState {
  std::array<bool, kNumLegs> in_contact{{true, true, true, true}};
  std::array<Vec3, kNumLegs> force_world{{Vec3::Zero(), Vec3::Zero(), Vec3::Zero(),
                                          Vec3::Zero()}};
  std::array<Vec3, kNumLegs> point_world{{Vec3::Zero(), Vec3::Zero(), Vec3::Zero(),
                                          Vec3::Zero()}};
};

struct ControlOutput {
  Vec16 tau = Vec16::Zero();
  Vec6 action6 = Vec6::Zero();
  std::array<Vec3, kNumLegs> force_des{{Vec3::Zero(), Vec3::Zero(), Vec3::Zero(),
                                        Vec3::Zero()}};
  ContactState contacts;
  double mpc_ms = 0.0;
  double mapper_ms = 0.0;
  int qp_status = 0;
  bool saturated = false;
  SafetyState safety = SafetyState::INIT;
  bool valid = false;
};

struct CentroidalState {
  double mass = 0.0;
  Vec3 com = Vec3::Zero();
  Vec3 com_vel = Vec3::Zero();
  Mat3 inertia_com = Mat3::Identity();
};

struct SegmentSpec {
  double x0 = 0.0;
  double x1 = 1.0;
  double amp = 0.0;
  double corridor_width = 0.5;
  double goal_tolerance = 0.15;
  double stop_speed_threshold = 0.2;
  double stop_duration = 0.3;
};

}  // namespace go2w
