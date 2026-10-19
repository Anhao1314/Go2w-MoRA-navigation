#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <string>

#include "fixture_io.hpp"
#include "go2w/dynamics.hpp"
#include "go2w/model.hpp"

namespace {

std::string fixture(const char* name) {
  return std::string(GO2W_FIXTURE_DIR) + "/" + name;
}

}  // namespace

TEST_CASE("Centroidal mass, COM and inertia match MuJoCo", "[dynamics][fixtures]") {
  const go2w::RobotModel model;
  const go2w::Dynamics dynamics(model);
  const auto rows = go2w::test::readNumericCsv(fixture("centroidal.csv"));
  REQUIRE(rows.size() == 50);

  double max_mass_error = 0.0;
  double max_com_error = 0.0;
  double max_inertia_error = 0.0;
  for (const auto& row : rows) {
    const go2w::RobotState state = go2w::test::stateFromRow(row);
    const go2w::CentroidalState result = dynamics.computeCentroidal(state);

    max_mass_error =
        std::max(max_mass_error, std::abs(result.mass - row[23]));
    const go2w::Vec3 expected_com(row[24], row[25], row[26]);
    max_com_error =
        std::max(max_com_error, (result.com - expected_com).cwiseAbs().maxCoeff());

    Eigen::Matrix3d expected_inertia;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        expected_inertia(r, c) = row[27 + r * 3 + c];
      }
    }
    max_inertia_error = std::max(
        max_inertia_error,
        (result.inertia_com - expected_inertia).cwiseAbs().maxCoeff());
  }

  INFO("mass error: " << max_mass_error);
  INFO("COM error: " << max_com_error);
  INFO("inertia error: " << max_inertia_error);
  REQUIRE(max_mass_error < 1e-9);
  REQUIRE(max_com_error < 1e-9);
  REQUIRE(max_inertia_error < 1e-8);
}

TEST_CASE("Total mass is the sum of link masses", "[dynamics]") {
  const go2w::RobotModel model;
  REQUIRE(std::abs(model.totalMass() - 19.126408) < 1e-9);
}
