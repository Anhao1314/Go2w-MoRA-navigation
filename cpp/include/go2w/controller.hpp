#pragma once

#include "go2w/dynamics.hpp"
#include "go2w/force_mapper.hpp"
#include "go2w/kinematics.hpp"
#include "go2w/model.hpp"
#include "go2w/path_tracker.hpp"
#include "go2w/safety.hpp"
#include "go2w/types.hpp"

namespace go2w {

struct S0Config {
  // P0 default: scripted teacher. MPC is validated standalone and reserved
  // for the P1 16-torque / ROS2 closed loop.
  bool use_mpc = false;
  PathTrackerConfig path;
  ForceMapperConfig mapper;
  SafetyConfig safety;
};

class S0Controller {
 public:
  explicit S0Controller(const S0Config& config = {});

  void reset(const RobotState& state);
  ControlOutput step(const RobotState& state, const HighLevelCommand& command);
  ControlOutput stepLegacy(const RobotState& state, double speed_scale,
                           double turn_adjust, int segment_idx,
                           double dist_to_subgoal, double speed);

  const S0Config& config() const { return config_; }
  const RobotModel& model() const { return model_; }

 private:
  S0Config config_;
  RobotModel model_;
  Kinematics kinematics_;
  Dynamics dynamics_;
  PathTracker path_;
  ForceMapper mapper_;
  SafetyLayer safety_;
};

}  // namespace go2w
