#include "go2w/controller.hpp"

#include <cmath>

namespace go2w {

S0Controller::S0Controller(const S0Config& config)
    : config_(config),
      model_(),
      kinematics_(model_),
      dynamics_(model_),
      path_(config_.path),
      mapper_(config_.mapper),
      safety_(config_.safety) {}

void S0Controller::reset(const RobotState& state) {
  (void)state;
  safety_.reset();
}

ControlOutput S0Controller::step(const RobotState& state,
                                 const HighLevelCommand& command) {
  ControlOutput out;
  const bool qp_ok = true;  // P0 scripted mode has no QP.
  out.safety = safety_.update(state, command, 0.01, qp_ok);

  if (command.mode != HighLevelCommand::Mode::LEGACY_6D) {
    out.valid = false;
    return out;
  }

  const double x = state.base.p.x();
  const double y = state.base.p.y();
  const Eigen::Matrix3d R = state.base.q.toRotationMatrix();
  const double yaw = std::atan2(R(1, 0), R(0, 0));

  const Vec6 action = path_.legacyAction(
      command.segment_idx, x, y, yaw, command.speed_scale, command.turn_adjust,
      command.dist_to_subgoal, command.speed);

  out.action6 = safety_.filterAction(action, out.safety);
  out.tau.setZero();
  for (int i = 0; i < 4; ++i) {
    out.tau[i + 12] = out.action6[i] * constants::kTorqueLimitWheel;
  }
  out.valid = true;
  return out;
}

ControlOutput S0Controller::stepLegacy(const RobotState& state,
                                       double speed_scale, double turn_adjust,
                                       int segment_idx, double dist_to_subgoal,
                                       double speed) {
  HighLevelCommand command;
  command.mode = HighLevelCommand::Mode::LEGACY_6D;
  command.speed_scale = speed_scale;
  command.turn_adjust = turn_adjust;
  command.segment_idx = segment_idx;
  command.dist_to_subgoal = dist_to_subgoal;
  command.speed = speed;
  return step(state, command);
}

}  // namespace go2w
