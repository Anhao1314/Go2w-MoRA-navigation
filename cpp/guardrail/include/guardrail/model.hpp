#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <string>
#include <vector>

namespace guardrail {

/// One rigid body of the robot, parameters exported from the simulator asset.
struct Body {
  std::string name;
  double mass = 0.0;
  /// Centre of mass expressed in the body frame.
  Eigen::Vector3d com_b = Eigen::Vector3d::Zero();
  /// Inertia tensor about the COM, expressed in the body frame.
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
};

/// A joint connecting two bodies.  Revolute joints carry an axis and limits;
/// fixed joints (feet, head) have `revolute == false` and are used only for
/// kinematics and mass distribution.
struct Joint {
  std::string name;
  int parent = -1;  ///< body index
  int child = -1;   ///< body index
  bool revolute = true;
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  double limit_min = 0.0;
  double limit_max = 0.0;
  double velocity_limit = 0.0;
  double effort_limit = 0.0;
  double stiffness = 0.0;
  double damping = 0.0;
  /// Joint frame pose in the parent body frame (USD localPos0/localRot0).
  Eigen::Isometry3d parent_to_joint = Eigen::Isometry3d::Identity();
  /// Joint frame pose in the child body frame (USD localPos1/localRot1).
  Eigen::Isometry3d child_to_joint = Eigen::Isometry3d::Identity();
};

/// Rigid-body model loaded from `locomotion/guardrail/go2_model.json`.
///
/// The model is deliberately data-driven: nothing about the Go2 link layout is
/// hard-coded here, so the same code works for another quadruped once its JSON
/// is exported.
class Model {
 public:
  /// Load and validate a model file.  Throws std::runtime_error on malformed input.
  static Model fromJson(const std::string& path);

  int numBodies() const { return static_cast<int>(bodies_.size()); }
  int numJoints() const { return static_cast<int>(joints_.size()); }
  const std::vector<Body>& bodies() const { return bodies_; }
  const std::vector<Joint>& joints() const { return joints_; }
  /// Indices (into joints_) of actuated joints, in joint-position order.
  const std::vector<int>& actuatedJoints() const { return actuated_; }
  /// Body indices of the feet, in `FL/FR/RL/RR` order as exported.
  const std::vector<int>& feet() const { return feet_; }
  const Eigen::VectorXd& defaultJointPos() const { return default_joint_pos_; }
  double totalMass() const { return total_mass_; }
  /// Joint indices whose parent is `body`.
  const std::vector<std::vector<int>>& jointsByParent() const { return by_parent_; }
  /// Index of the base body (root of the tree).
  int rootBody() const { return root_body_; }

 private:
  std::vector<Body> bodies_;
  std::vector<Joint> joints_;
  std::vector<int> actuated_;
  std::vector<int> feet_;
  std::vector<std::vector<int>> by_parent_;
  Eigen::VectorXd default_joint_pos_;
  double total_mass_ = 0.0;
  int root_body_ = 0;
};

}  // namespace guardrail
