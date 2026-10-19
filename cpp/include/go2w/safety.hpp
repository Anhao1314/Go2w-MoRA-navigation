#pragma once

#include "go2w/types.hpp"

namespace go2w {

struct SafetyConfig {
  double max_roll = 0.5;
  double max_pitch = 0.5;
  double min_height = 0.15;
  double qp_fail_time = 0.1;
  double command_timeout = 0.25;
};

class SafetyLayer {
 public:
  explicit SafetyLayer(const SafetyConfig& config = {});

  void reset();
  SafetyState update(const RobotState& state, const HighLevelCommand& command,
                     double dt, bool qp_ok);
  Vec6 filterAction(const Vec6& action, SafetyState state) const;
  SafetyState state() const { return state_; }

 private:
  SafetyConfig config_;
  SafetyState state_ = SafetyState::INIT;
  double qp_fail_duration_ = 0.0;
  double command_age_ = 0.0;
};

}  // namespace go2w
