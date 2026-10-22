#include "guardrail/model.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace guardrail {
namespace {

using nlohmann::json;

Eigen::Vector3d toVec3(const json& value, const char* field) {
  if (!value.is_array() || value.size() != 3) {
    throw std::runtime_error(std::string("expected a 3-vector for ") + field);
  }
  return Eigen::Vector3d(value[0].get<double>(), value[1].get<double>(),
                         value[2].get<double>());
}

/// USD stores quaternions as [real, i, j, k] (w, x, y, z).
Eigen::Quaterniond toQuat(const json& value, const char* field) {
  if (!value.is_array() || value.size() != 4) {
    throw std::runtime_error(std::string("expected a quaternion for ") + field);
  }
  return Eigen::Quaterniond(value[0].get<double>(), value[1].get<double>(),
                            value[2].get<double>(), value[3].get<double>())
      .normalized();
}

Eigen::Isometry3d toPose(const json& pos, const json& rot, const char* field) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = toVec3(pos, field);
  pose.linear() = toQuat(rot, field).toRotationMatrix();
  return pose;
}

int findBody(const std::vector<Body>& bodies, const std::string& name) {
  for (int index = 0; index < static_cast<int>(bodies.size()); ++index) {
    if (bodies[index].name == name) {
      return index;
    }
  }
  return -1;
}

}  // namespace

Model Model::fromJson(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open model file: " + path);
  }
  json data;
  stream >> data;

  Model model;

  for (const auto& entry : data.at("bodies")) {
    Body body;
    body.name = entry.at("name").get<std::string>();
    body.mass = entry.at("mass").get<double>();
    body.com_b = toVec3(entry.at("com_b"), "com_b");
    const auto& inertia = entry.at("inertia");
    if (!inertia.is_array() || inertia.size() != 9) {
      throw std::runtime_error("expected a 9-element inertia for " + body.name);
    }
    Eigen::Matrix3d matrix;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        matrix(row, col) = inertia[row * 3 + col].get<double>();
      }
    }
    body.inertia = matrix;
    model.total_mass_ += body.mass;
    model.bodies_.push_back(std::move(body));
  }
  if (model.bodies_.empty()) {
    throw std::runtime_error("model has no bodies");
  }

  // root: the body that never appears as a child
  {
    std::vector<bool> is_child(model.bodies_.size(), false);
    for (const auto& entry : data.at("joints")) {
      const std::string child = entry.at("body1").get<std::string>();
      const int index = findBody(model.bodies_, child);
      if (index >= 0) {
        is_child[static_cast<size_t>(index)] = true;
      }
    }
    model.root_body_ = 0;
    for (int index = 0; index < static_cast<int>(is_child.size()); ++index) {
      if (!is_child[static_cast<size_t>(index)]) {
        model.root_body_ = index;
        break;
      }
    }
  }

  auto addJoint = [&model](const json& entry, bool revolute) {
    Joint joint;
    joint.name = entry.at("name").get<std::string>();
    joint.revolute = revolute;
    joint.parent = findBody(model.bodies_, entry.at("body0").get<std::string>());
    joint.child = findBody(model.bodies_, entry.at("body1").get<std::string>());
    if (joint.parent < 0 || joint.child < 0) {
      throw std::runtime_error("joint references an unknown body: " + joint.name);
    }
    joint.parent_to_joint = toPose(entry.at("local_pos0"), entry.at("local_rot0"), "local_pos0");
    joint.child_to_joint = toPose(entry.at("local_pos1"), entry.at("local_rot1"), "local_pos1");
    if (revolute) {
      const auto& axis = entry.at("axis");
      joint.axis = toVec3(axis, "axis");
      if (joint.axis.norm() < 1e-9) {
        throw std::runtime_error("zero joint axis: " + joint.name);
      }
      joint.axis.normalize();
      const auto& limit = entry.at("limit");
      joint.limit_min = limit.at(0).get<double>();
      joint.limit_max = limit.at(1).get<double>();
      joint.velocity_limit = entry.value("velocity_limit", 0.0);
      joint.effort_limit = entry.value("effort_limit", 0.0);
      joint.stiffness = entry.value("stiffness", 0.0);
      joint.damping = entry.value("damping", 0.0);
    }
    model.joints_.push_back(std::move(joint));
  };

  for (const auto& entry : data.at("joints")) {
    addJoint(entry, true);
  }
  if (data.contains("fixed_joints")) {
    for (const auto& entry : data.at("fixed_joints")) {
      addJoint(entry, false);
    }
  }

  model.by_parent_.assign(model.bodies_.size(), {});
  for (int index = 0; index < static_cast<int>(model.joints_.size()); ++index) {
    model.by_parent_[static_cast<size_t>(model.joints_[static_cast<size_t>(index)].parent)].push_back(index);
    if (model.joints_[static_cast<size_t>(index)].revolute) {
      model.actuated_.push_back(index);
    }
  }

  for (const auto& name : data.value("feet", std::vector<std::string>{})) {
    const int index = findBody(model.bodies_, name);
    if (index >= 0) {
      model.feet_.push_back(index);
    }
  }

  if (data.contains("default_joint_pos")) {
    const auto& values = data.at("default_joint_pos");
    model.default_joint_pos_ = Eigen::VectorXd(static_cast<Eigen::Index>(values.size()));
    for (size_t index = 0; index < values.size(); ++index) {
      model.default_joint_pos_[static_cast<Eigen::Index>(index)] = values[index].get<double>();
    }
    if (model.default_joint_pos_.size() != static_cast<Eigen::Index>(model.actuated_.size())) {
      throw std::runtime_error("default_joint_pos does not match the actuated joint count");
    }
  } else {
    model.default_joint_pos_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(model.actuated_.size()));
  }

  return model;
}

}  // namespace guardrail
