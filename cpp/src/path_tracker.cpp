#include "go2w/path_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace go2w {
namespace {

constexpr double kPi = 3.14159265358979323846;

double wrapAngle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

double clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

}  // namespace

PathTracker::PathTracker(const PathTrackerConfig& config) : config_(config) {}

double PathTracker::centerY(int segment_idx, double x) const {
  const int idx = std::max(0, std::min<int>(1, segment_idx));
  const auto& seg = config_.segments[idx];
  const double length = seg.x1 - seg.x0;
  if (length <= 0.0) {
    return 0.0;
  }
  // The Python reference has ya == 0 for segment 1 (sin(2*pi) == 0); keeping
  // the same expression here preserves bit-level parity with the teacher.
  return seg.amp * std::sin(2.0 * kPi * (x - seg.x0) / length);
}

double PathTracker::teacherBias(int segment_idx, double x, double y,
                                double yaw) const {
  const int idx = std::max(0, std::min<int>(1, segment_idx));
  const auto& seg = config_.segments[idx];
  const double xt = std::min(x + config_.lookahead, seg.x1);
  const double yt = centerY(idx, xt);
  const double desired_yaw = std::atan2(yt - y, xt - x);
  const double ang_err = wrapAngle(desired_yaw - yaw);
  const double dev = y - centerY(idx, x);
  return clamp(-config_.k_ang * ang_err + config_.k_dev * dev,
               -config_.bias_clip, config_.bias_clip);
}

double PathTracker::referenceYaw(int segment_idx, double x, double y) const {
  const int idx = std::max(0, std::min<int>(1, segment_idx));
  const auto& seg = config_.segments[idx];
  const double xt = std::min(x + config_.lookahead, seg.x1);
  const double yt = centerY(idx, xt);
  return std::atan2(yt - y, xt - x);
}

double PathTracker::referenceSpeed(double speed_scale) const {
  // Conservative mapping used only by the MPC benchmark. The scripted mode
  // does not use it.
  return clamp(0.5 * speed_scale, 0.1, 0.8);
}

Vec6 PathTracker::legacyAction(int segment_idx, double x, double y, double yaw,
                               double speed_scale, double turn_adjust,
                               double dist_to_subgoal, double speed) const {
  const int idx = std::max(0, std::min<int>(1, segment_idx));
  const auto& seg = config_.segments[idx];
  double scale = clamp(speed_scale, 0.5, 1.5);
  const double adjust = clamp(turn_adjust, -0.5, 0.5);
  if (config_.pre_brake) {
    if (dist_to_subgoal < 0.3) {
      scale = std::min(scale, 0.4);
    } else if (dist_to_subgoal < 0.8) {
      scale = std::min(scale, 0.3 + (dist_to_subgoal - 0.3) / 0.5 * 0.7);
    }
  }

  if (dist_to_subgoal <= seg.goal_tolerance) {
    if (std::abs(speed) < seg.stop_speed_threshold) {
      return Vec6::Zero();
    }
    const double brake = -0.02 * std::abs(speed);
    return Vec6(brake, brake, brake, brake, 0.0, 0.0);
  }

  const double bias =
      clamp(teacherBias(idx, x, y, yaw) + adjust, -config_.bias_clip,
            config_.bias_clip);
  const double forward = config_.forward * scale;
  Vec6 action;
  action << forward - bias, forward + bias, forward - bias, forward + bias,
      0.0, 0.0;
  return action.cwiseMax(-1.0).cwiseMin(1.0);
}

}  // namespace go2w
