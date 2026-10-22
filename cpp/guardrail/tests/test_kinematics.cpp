#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "guardrail/kinematics.hpp"
#include "guardrail/model.hpp"

#ifndef GUARDRAIL_MODEL_PATH
#error "GUARDRAIL_MODEL_PATH must point at the exported go2_model.json"
#endif

namespace {

guardrail::Model loadModel() { return guardrail::Model::fromJson(GUARDRAIL_MODEL_PATH); }

int findBody(const guardrail::Model& model, const std::string& name) {
  for (int index = 0; index < model.numBodies(); ++index) {
    if (model.bodies()[static_cast<size_t>(index)].name == name) {
      return index;
    }
  }
  return -1;
}

}  // namespace

TEST_CASE("exported Go2 model loads with the expected shape", "[model]") {
  const auto model = loadModel();

  REQUIRE(model.numBodies() == 19);
  REQUIRE(static_cast<int>(model.actuatedJoints().size()) == 12);
  REQUIRE(model.feet().size() == 4);

  // Go2 datasheet mass is about 15 kg
  REQUIRE(model.totalMass() == Catch::Approx(14.971).margin(0.5));

  // every actuated joint must carry an effort limit from the actuator config
  for (const int index : model.actuatedJoints()) {
    const auto& joint = model.joints()[static_cast<size_t>(index)];
    REQUIRE(joint.revolute);
    REQUIRE(joint.axis.norm() == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(joint.effort_limit > 0.0);
    REQUIRE(joint.limit_min <= joint.limit_max);
  }
}

TEST_CASE("forward kinematics is symmetric between left and right legs", "[fk]") {
  const auto model = loadModel();
  const Eigen::Isometry3d root = Eigen::Isometry3d::Identity();
  const auto fk = guardrail::forwardKinematics(model, root, model.defaultJointPos());

  const int fl_hip = findBody(model, "FL_hip");
  const int fr_hip = findBody(model, "FR_hip");
  const int fl_thigh = findBody(model, "FL_thigh");
  const int fr_thigh = findBody(model, "FR_thigh");
  REQUIRE(fl_hip >= 0);
  REQUIRE(fr_hip >= 0);

  const Eigen::Vector3d hip_l = fk.world[static_cast<size_t>(fl_hip)].translation();
  const Eigen::Vector3d hip_r = fk.world[static_cast<size_t>(fr_hip)].translation();
  CHECK(hip_l.x() == Catch::Approx(hip_r.x()).margin(1e-9));
  CHECK(hip_l.y() == Catch::Approx(-hip_r.y()).margin(1e-9));
  CHECK(hip_l.z() == Catch::Approx(hip_r.z()).margin(1e-9));

  const Eigen::Vector3d thigh_l = fk.world[static_cast<size_t>(fl_thigh)].translation();
  const Eigen::Vector3d thigh_r = fk.world[static_cast<size_t>(fr_thigh)].translation();
  CHECK(thigh_l.y() == Catch::Approx(-thigh_r.y()).margin(1e-9));
  CHECK(thigh_l.z() == Catch::Approx(thigh_r.z()).margin(1e-9));

  // the base is the only body that is always reachable
  CHECK(fk.valid[static_cast<size_t>(model.rootBody())]);
  CHECK(fk.mass_covered > 0.0);
}

TEST_CASE("forward kinematics stays finite at joint limits", "[fk]") {
  const auto model = loadModel();
  const Eigen::Isometry3d root = Eigen::Isometry3d::Identity();

  Eigen::VectorXd low = model.defaultJointPos();
  Eigen::VectorXd high = model.defaultJointPos();
  for (int slot = 0; slot < static_cast<int>(low.size()); ++slot) {
    const auto& joint = model.joints()[static_cast<size_t>(model.actuatedJoints()[static_cast<size_t>(slot)])];
    low[slot] = joint.limit_min;
    high[slot] = joint.limit_max;
  }

  for (const auto& q : {low, high}) {
    const auto fk = guardrail::forwardKinematics(model, root, q);
    for (size_t index = 0; index < fk.world.size(); ++index) {
      if (!fk.valid[index]) {
        continue;
      }
      REQUIRE(fk.world[index].matrix().allFinite());
    }
    REQUIRE(fk.com_world.allFinite());
    REQUIRE(fk.mass_covered > 0.0);
  }
}

TEST_CASE("forward kinematics rejects a wrong joint-vector size", "[fk]") {
  const auto model = loadModel();
  Eigen::VectorXd q(model.actuatedJoints().size() - 1);
  q.setZero();
  REQUIRE_THROWS_AS(
      guardrail::forwardKinematics(model, Eigen::Isometry3d::Identity(), q),
      std::invalid_argument);
}

TEST_CASE("convex hull of a square keeps its four corners", "[polygon]") {
  std::vector<Eigen::Vector2d> points = {
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.5, 0.5}};
  const auto hull = guardrail::convexHull(points);
  REQUIRE(hull.size() == 4);

  guardrail::SupportPolygon polygon;
  polygon.vertices = hull;
  CHECK(polygon.contains(Eigen::Vector2d(0.5, 0.5)));
  CHECK(polygon.contains(Eigen::Vector2d(0.01, 0.01)));
  CHECK_FALSE(polygon.contains(Eigen::Vector2d(1.5, 0.5)));
  CHECK(polygon.margin(Eigen::Vector2d(0.5, 0.5)) > 0.0);
  CHECK(polygon.margin(Eigen::Vector2d(2.0, 0.5)) < 0.0);
}

TEST_CASE("support polygon is empty while feet frames are not exported", "[polygon]") {
  // The exporter does not yet emit the fixed foot joints, so feet are not
  // reachable from the root.  This test pins the current, honest behaviour and
  // must be flipped to a real containment test once fixed joints ship.
  const auto model = loadModel();
  const auto fk =
      guardrail::forwardKinematics(model, Eigen::Isometry3d::Identity(), model.defaultJointPos());
  const auto polygon = guardrail::supportPolygon(model, fk);
  bool any_foot_valid = false;
  for (const int foot : model.feet()) {
    any_foot_valid = any_foot_valid || fk.valid[static_cast<size_t>(foot)];
  }
  if (!any_foot_valid) {
    REQUIRE(polygon.empty());
  } else {
    REQUIRE_FALSE(polygon.empty());
    // the centre of mass must sit above the support polygon in a standing pose
    CHECK(polygon.contains(Eigen::Vector2d(fk.com_world.x(), fk.com_world.y())));
  }
}
