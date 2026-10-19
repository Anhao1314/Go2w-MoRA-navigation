#include "go2w/safety.hpp"

#include <algorithm>
#include <cmath>

namespace go2w {

SafetyLayer::SafetyLayer(const SafetyConfig& config) : config_(config) {}

void SafetyLayer::reset() {
  state_ = SafetyState::INIT;
  qp_fail_duration_ = 0.0;
  command_age_ = 0.0;
}

SafetyState SafetyLayer::update(const RobotState& state,
                                const HighLevelCommand& command, double dt,
                                bool qp_ok) {
  const Eigen::Matrix3d R = state.base.q.toRotationMatrix();
  const double roll = std::atan2(R(2, 1), R(2, 2));
  const double pitch = std::atan2(-R(2, 0), std::sqrt(R(2, 1) * R(2, 1) +
                                                       R(2, 2) * R(2, 2)));

  if (!std::isfinite(state.base.p.z()) || state.base.p.z() < config_.min_height) {
    state_ = SafetyState::ESTOP;
    return state_;
  }
  if (std::abs(roll) > config_.max_roll || std::abs(pitch) > config_.max_pitch) {
    state_ = SafetyState::SAFE_STOP;
    return state_;
  }

  command_age_ += dt;
  if (command_age_ > config_.command_timeout) {
    state_ = SafetyState::DEGRADED;
    return state_;
  }
  command_age_ = 0.0;

  if (!qp_ok) {
    qp_fail_duration_ += dt;
    state_ = (qp_fail_duration_ > config_.qp_fail_time)
                 ? SafetyState::DEGRADED
                 : SafetyState::TRACK;
    return state_;
  }
  qp_fail_duration_ = 0.0;

  if (command.mode == HighLevelCommand::Mode::SAFE_STOP) {
    state_ = SafetyState::SAFE_STOP;
  } else {
    state_ = SafetyState::TRACK;
  }
  return state_;
}

Vec6 SafetyLayer::filterAction(const Vec6& action, SafetyState state) const {
  if (state == SafetyState::SAFE_STOP || state == SafetyState::ESTOP ||
      state == SafetyState::INIT || state == SafetyState::STAND) {
    return Vec6::Zero();
  }
  double scale = 1.0;
  if (state == SafetyState::DEGRADED) {
    scale = 0.5;
  }
  return (scale * action).cwiseMax(-1.0).cwiseMin(1.0);
}

}  // namespace go2w
