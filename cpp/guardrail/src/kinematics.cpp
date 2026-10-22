#include "guardrail/kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace guardrail {

FkResult forwardKinematics(const Model& model, const Eigen::Isometry3d& root_pose,
                           const Eigen::VectorXd& q) {
  const int num_bodies = model.numBodies();
  const int num_joints = model.numJoints();
  const int num_actuated = static_cast<int>(model.actuatedJoints().size());
  if (q.size() != num_actuated) {
    throw std::invalid_argument("joint position vector has the wrong size");
  }

  // map joint index -> position index in q
  std::vector<int> q_index(static_cast<size_t>(num_joints), -1);
  for (int slot = 0; slot < num_actuated; ++slot) {
    q_index[static_cast<size_t>(model.actuatedJoints()[static_cast<size_t>(slot)])] = slot;
  }

  FkResult result;
  result.world.assign(static_cast<size_t>(num_bodies), Eigen::Isometry3d::Identity());
  result.valid.assign(static_cast<size_t>(num_bodies), false);

  const int root = model.rootBody();
  if (root < 0 || root >= num_bodies) {
    throw std::runtime_error("model has no valid root body");
  }
  result.world[static_cast<size_t>(root)] = root_pose;
  result.valid[static_cast<size_t>(root)] = true;

  std::queue<int> frontier;
  frontier.push(root);
  while (!frontier.empty()) {
    const int parent = frontier.front();
    frontier.pop();
    for (const int joint_index : model.jointsByParent()[static_cast<size_t>(parent)]) {
      const Joint& joint = model.joints()[static_cast<size_t>(joint_index)];
      Eigen::Isometry3d rotation = Eigen::Isometry3d::Identity();
      if (joint.revolute) {
        const int slot = q_index[static_cast<size_t>(joint_index)];
        if (slot < 0) {
          continue;
        }
        rotation.linear() =
            Eigen::AngleAxisd(q[slot], joint.axis).toRotationMatrix();
      }
      result.world[static_cast<size_t>(joint.child)] =
          result.world[static_cast<size_t>(parent)] * joint.parent_to_joint * rotation *
          joint.child_to_joint.inverse();
      result.valid[static_cast<size_t>(joint.child)] = true;
      frontier.push(joint.child);
    }
  }

  Eigen::Vector3d weighted = Eigen::Vector3d::Zero();
  double mass = 0.0;
  for (int index = 0; index < num_bodies; ++index) {
    if (!result.valid[static_cast<size_t>(index)]) {
      continue;
    }
    const Body& body = model.bodies()[static_cast<size_t>(index)];
    weighted += body.mass * (result.world[static_cast<size_t>(index)] * body.com_b);
    mass += body.mass;
  }
  if (mass > 0.0) {
    result.com_world = weighted / mass;
  }
  result.mass_covered = mass;
  return result;
}

std::vector<Eigen::Vector2d> convexHull(std::vector<Eigen::Vector2d> points) {
  if (points.size() <= 2) {
    return points;
  }
  std::sort(points.begin(), points.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    if (a.x() != b.x()) {
      return a.x() < b.x();
    }
    return a.y() < b.y();
  });
  points.erase(std::unique(points.begin(), points.end()), points.end());
  if (points.size() <= 2) {
    return points;
  }

  auto cross = [](const Eigen::Vector2d& o, const Eigen::Vector2d& a,
                  const Eigen::Vector2d& b) {
    return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
  };

  std::vector<Eigen::Vector2d> hull(2 * points.size());
  size_t k = 0;
  for (size_t index = 0; index < points.size(); ++index) {
    while (k >= 2 && cross(hull[k - 2], hull[k - 1], points[index]) <= 0.0) {
      --k;
    }
    hull[k++] = points[index];
  }
  const size_t lower = k + 1;
  for (size_t index = points.size() - 1; index-- > 0;) {
    while (k >= lower && cross(hull[k - 2], hull[k - 1], points[index]) <= 0.0) {
      --k;
    }
    hull[k++] = points[index];
  }
  hull.resize(k - 1);  // last point equals the first

  // counter-clockwise by construction
  return hull;
}

double SupportPolygon::margin(const Eigen::Vector2d& point) const {
  if (vertices.size() < 3) {
    return -std::numeric_limits<double>::infinity();
  }
  double best = std::numeric_limits<double>::infinity();
  for (size_t index = 0; index < vertices.size(); ++index) {
    const Eigen::Vector2d& a = vertices[index];
    const Eigen::Vector2d& b = vertices[(index + 1) % vertices.size()];
    Eigen::Vector2d edge = b - a;
    const double length = edge.norm();
    if (length < 1e-12) {
      continue;
    }
    // inward normal for a counter-clockwise polygon
    const Eigen::Vector2d inward(-edge.y() / length, edge.x() / length);
    best = std::min(best, (point - a).dot(inward));
  }
  return best;
}

SupportPolygon supportPolygon(const Model& model, const FkResult& fk,
                              double foot_height_tolerance) {
  std::vector<size_t> feet;
  double lowest = std::numeric_limits<double>::infinity();
  for (const int index : model.feet()) {
    const size_t slot = static_cast<size_t>(index);
    if (slot >= fk.world.size() || !fk.valid[slot]) {
      continue;
    }
    feet.push_back(slot);
    lowest = std::min(lowest, fk.world[slot].translation().z());
  }

  SupportPolygon polygon;
  if (feet.empty()) {
    return polygon;
  }

  std::vector<Eigen::Vector2d> points;
  for (const size_t slot : feet) {
    const Eigen::Vector3d position = fk.world[slot].translation();
    if (position.z() <= lowest + foot_height_tolerance) {
      points.emplace_back(position.x(), position.y());
    }
  }
  polygon.vertices = convexHull(std::move(points));
  return polygon;
}

}  // namespace guardrail
