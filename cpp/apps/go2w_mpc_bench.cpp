#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "go2w/dynamics.hpp"
#include "go2w/kinematics.hpp"
#include "go2w/model.hpp"
#include "go2w/mpc.hpp"

namespace {

go2w::MpcInput makeStandInput() {
  const go2w::RobotModel model;
  const go2w::Kinematics kinematics(model);
  const go2w::Dynamics dynamics(model);

  go2w::RobotState state;
  state.base.p = go2w::Vec3(0.0, 0.0, go2w::constants::kStandBaseHeight);
  state.base.q = Eigen::Quaterniond::Identity();
  for (int leg = 0; leg < go2w::kNumLegs; ++leg) {
    state.joints.q_leg.segment<3>(3 * leg) =
        go2w::Vec3(go2w::constants::kStandLeg[0],
                   go2w::constants::kStandLeg[1],
                   go2w::constants::kStandLeg[2]);
  }
  const auto transforms = kinematics.computeBodyTransforms(state);
  const auto centroidal = dynamics.computeCentroidal(state);

  go2w::MpcInput input;
  input.mass = centroidal.mass;
  input.com = centroidal.com;
  input.inertia_com = centroidal.inertia_com;
  input.z_ref = centroidal.com.z();
  for (int leg = 0; leg < go2w::kNumLegs; ++leg) {
    const int body = model.wheelBodyIndex(static_cast<go2w::Leg>(leg));
    input.contact_points_world[leg] =
        transforms.T[body].translation() -
        go2w::Vec3(0.0, 0.0, go2w::constants::kWheelRadius);
  }
  return input;
}

}  // namespace

int main() {
  const go2w::MpcInput input = makeStandInput();
  go2w::SrbdMpc mpc;
  constexpr int kIterations = 200;
  std::vector<double> times;
  times.reserve(kIterations);
  int solved = 0;
  double total_vertical_force = 0.0;
  for (int i = 0; i < kIterations; ++i) {
    const go2w::MpcOutput output = mpc.solve(input);
    times.push_back(output.solve_ms);
    if (output.success) {
      ++solved;
      if (i == 0) {
        for (const auto& force : output.forces) {
          total_vertical_force += force.z();
        }
      }
    }
  }
  std::sort(times.begin(), times.end());
  const auto percentile = [&times](double p) {
    const std::size_t index = static_cast<std::size_t>(
        std::min<double>(times.size() - 1, p * (times.size() - 1)));
    return times[index];
  };
  const double mean =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  const double weight = input.mass * 9.81;
  std::printf(
      "{\n"
      "  \"iterations\": %d,\n"
      "  \"solved\": %d,\n"
      "  \"mean_ms\": %.6f,\n"
      "  \"p50_ms\": %.6f,\n"
      "  \"p95_ms\": %.6f,\n"
      "  \"p99_ms\": %.6f,\n"
      "  \"max_ms\": %.6f,\n"
      "  \"horizontal_hover_vertical_force_n\": %.6f,\n"
      "  \"weight_n\": %.6f,\n"
      "  \"vertical_force_error_pct\": %.6f\n"
      "}\n",
      kIterations, solved, mean, percentile(0.50), percentile(0.95),
      percentile(0.99), times.back(), total_vertical_force, weight,
      100.0 * std::abs(total_vertical_force - weight) / weight);
  return solved == kIterations ? 0 : 1;
}
