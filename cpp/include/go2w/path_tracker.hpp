#pragma once

#include <array>

#include "go2w/types.hpp"

namespace go2w {

struct PathTrackerConfig {
  double lookahead = 0.7;
  double k_ang = 2.5;
  double k_dev = 0.5;
  double bias_clip = 1.0;
  double forward = 0.12;
  bool pre_brake = false;
  std::array<SegmentSpec, 2> segments{{
      {0.8, 4.3, 0.12, 0.50, 0.15, 0.2, 0.3},
      {4.3, 10.8, 0.35, 0.40, 0.15, 0.2, 0.3},
  }};
};

class PathTracker {
 public:
  explicit PathTracker(const PathTrackerConfig& config);

  double centerY(int segment_idx, double x) const;
  double teacherBias(int segment_idx, double x, double y, double yaw) const;
  double referenceYaw(int segment_idx, double x, double y) const;
  double referenceSpeed(double speed_scale) const;

  // Exact port of MultiSegmentLowLevelController.compute_action.
  Vec6 legacyAction(int segment_idx, double x, double y, double yaw,
                    double speed_scale, double turn_adjust,
                    double dist_to_subgoal, double speed) const;

  const PathTrackerConfig& config() const { return config_; }

 private:
  PathTrackerConfig config_;
};

}  // namespace go2w
