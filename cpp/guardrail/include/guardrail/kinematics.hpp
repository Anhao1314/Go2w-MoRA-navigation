#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

#include "guardrail/model.hpp"

namespace guardrail {

/// Result of a floating-base forward-kinematics pass.
struct FkResult {
  /// World pose of each body, indexed like `Model::bodies()`.
  std::vector<Eigen::Isometry3d> world;
  /// False for bodies not reachable from the root through the exported joints
  /// (for example feet while their fixed joints are not exported yet).
  std::vector<bool> valid;
  /// Mass-weighted centre of mass over the valid bodies.
  Eigen::Vector3d com_world = Eigen::Vector3d::Zero();
  /// Sum of masses that contributed to `com_world` (kg).
  double mass_covered = 0.0;
};

/// Floating-base forward kinematics.
///
/// `root_pose` places the base body in the world; `q` holds joint positions in
/// `Model::actuatedJoints()` order.  The joint transform follows the USD
/// convention: ``world_child = world_parent * localPos0/localRot0 *
/// Rot(axis, q) * inverse(localPos1/localRot1)``.
FkResult forwardKinematics(const Model& model, const Eigen::Isometry3d& root_pose,
                           const Eigen::VectorXd& q);

/// Convex support polygon in the ground plane, vertices counter-clockwise.
struct SupportPolygon {
  std::vector<Eigen::Vector2d> vertices;

  bool empty() const { return vertices.size() < 3; }

  /// Signed distance from `point` to the polygon boundary; positive inside.
  /// Returns -infinity for a degenerate polygon and +infinity for a single
  /// point inside nothing (see tests for the exact conventions).
  double margin(const Eigen::Vector2d& point) const;

  bool contains(const Eigen::Vector2d& point) const { return margin(point) >= 0.0; }
};

/// Build the support polygon from the feet currently near the lowest foot.
///
/// `foot_height_tolerance` (m) selects which feet count as supporting: feet
/// whose z is within that distance of the lowest foot.
SupportPolygon supportPolygon(const Model& model, const FkResult& fk,
                              double foot_height_tolerance = 0.03);

/// Android monotone-chain convex hull; exposed for tests.
std::vector<Eigen::Vector2d> convexHull(std::vector<Eigen::Vector2d> points);

}  // namespace guardrail
