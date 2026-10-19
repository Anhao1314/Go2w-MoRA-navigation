#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include "go2w/mpc.hpp"

#include <Eigen/Sparse>
#include <osqp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#if defined(_WIN32)
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

namespace go2w {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Eigen::Matrix3d skew(const Vec3& v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

struct CscOwner {
  csc* matrix = nullptr;
  ~CscOwner() {
    if (matrix != nullptr) {
      c_free(matrix->x);
      c_free(matrix->i);
      c_free(matrix->p);
      c_free(matrix);
    }
  }
};

csc* eigenToCsc(const Eigen::SparseMatrix<double>& input) {
  Eigen::SparseMatrix<double> matrix = input;
  matrix.makeCompressed();
  const c_int nz = static_cast<c_int>(matrix.nonZeros());
  auto* x = static_cast<c_float*>(c_malloc(sizeof(c_float) * nz));
  auto* i = static_cast<c_int*>(c_malloc(sizeof(c_int) * nz));
  auto* p = static_cast<c_int*>(
      c_malloc(sizeof(c_int) * (matrix.cols() + 1)));
  for (c_int k = 0; k < nz; ++k) {
    x[k] = matrix.valuePtr()[k];
    i[k] = static_cast<c_int>(matrix.innerIndexPtr()[k]);
  }
  for (int j = 0; j <= matrix.cols(); ++j) {
    p[j] = static_cast<c_int>(matrix.outerIndexPtr()[j]);
  }
  return csc_matrix(static_cast<c_int>(matrix.rows()),
                    static_cast<c_int>(matrix.cols()), nz, x, i, p);
}

}  // namespace

SrbdMpc::SrbdMpc(const MpcConfig& config) : config_(config) {}

MpcOutput SrbdMpc::solve(const MpcInput& input) {
  const auto start = std::chrono::steady_clock::now();
  MpcOutput output;
  const int N = std::max(1, config_.horizon);
  const int nx = 12;
  const int nu = 12;
  const int ns = 3;  // slack on p_z, roll, pitch
  const int n = nx * (N + 1) + nu * N + ns * N;
  const int n_state_bounds = 12 + nx * N + 24 * N + 6 * ns * N + ns * N;

  const auto x_index = [](int k) { return 12 * k; };
  const auto u_index = [N](int k) { return 12 * (N + 1) + 12 * k; };
  const auto s_index = [N](int k) {
    return 12 * (N + 1) + 12 * N + 3 * k;
  };

  // Linearized SRBD dynamics.
  Eigen::Matrix<double, nx, nx> A_dyn =
      Eigen::Matrix<double, nx, nx>::Identity();
  A_dyn.block<3, 3>(0, 3) = config_.dt * Eigen::Matrix3d::Identity();
  A_dyn.block<3, 3>(6, 9) = config_.dt * Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, nx, nu> B_dyn =
      Eigen::Matrix<double, nx, nu>::Zero();
  const Eigen::Matrix3d inertia_inv =
      input.inertia_com.inverse();
  for (int i = 0; i < kNumLegs; ++i) {
    B_dyn.block<3, 3>(3, 3 * i) =
        (config_.dt / input.mass) * Eigen::Matrix3d::Identity();
    B_dyn.block<3, 3>(9, 3 * i) =
        config_.dt * inertia_inv *
        skew(input.contact_points_world[i] - input.com);
  }

  Eigen::Matrix<double, nx, 1> c = Eigen::Matrix<double, nx, 1>::Zero();
  c[5] = -9.81 * config_.dt;

  Eigen::VectorXd x_meas(nx);
  x_meas << input.com, input.com_vel, input.theta, input.omega;

  Eigen::SparseMatrix<double> P(n, n);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  std::vector<Eigen::Triplet<double>> p_triplets;
  p_triplets.reserve(n);

  for (int k = 0; k <= N; ++k) {
    const auto& weights = (k == N) ? config_.q_terminal : config_.q;
    for (int j = 0; j < nx; ++j) {
      p_triplets.emplace_back(x_index(k) + j, x_index(k) + j,
                              2.0 * weights[j]);
    }
  }
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < nu; ++j) {
      p_triplets.emplace_back(u_index(k) + j, u_index(k) + j,
                              2.0 * config_.r[j]);
    }
    for (int j = 0; j < ns; ++j) {
      p_triplets.emplace_back(s_index(k) + j, s_index(k) + j,
                              2.0 * config_.slack_penalty);
    }
  }
  P.setFromTriplets(p_triplets.begin(), p_triplets.end());

  for (int k = 0; k <= N; ++k) {
    const auto& weights = (k == N) ? config_.q_terminal : config_.q;
    const double t = config_.dt * k;
    Eigen::Matrix<double, nx, 1> x_ref;
    x_ref << input.com.x() + input.v_ref.x() * t,
        input.com.y() + input.v_ref.y() * t, input.z_ref,
        input.v_ref.x(), input.v_ref.y(), 0.0,
        0.0, 0.0, input.yaw_ref,
        0.0, 0.0, input.wz_ref;
    for (int j = 0; j < nx; ++j) {
      q[x_index(k) + j] = -2.0 * weights[j] * x_ref[j];
    }
  }

  std::vector<Eigen::Triplet<double>> a_triplets;
  std::vector<double> lower;
  std::vector<double> upper;
  lower.reserve(n_state_bounds);
  upper.reserve(n_state_bounds);

  int row = 0;
  // Initial state equality.
  for (int j = 0; j < nx; ++j) {
    a_triplets.emplace_back(row, x_index(0) + j, 1.0);
    lower.push_back(x_meas[j]);
    upper.push_back(x_meas[j]);
    ++row;
  }

  // Dynamics: x_{k+1} - A x_k - B u_k = c.
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < nx; ++j) {
      a_triplets.emplace_back(row, x_index(k + 1) + j, 1.0);
      for (int jj = 0; jj < nx; ++jj) {
        if (A_dyn(j, jj) != 0.0) {
          a_triplets.emplace_back(row, x_index(k) + jj, -A_dyn(j, jj));
        }
      }
      for (int jj = 0; jj < nu; ++jj) {
        if (B_dyn(j, jj) != 0.0) {
          a_triplets.emplace_back(row, u_index(k) + jj, -B_dyn(j, jj));
        }
      }
      lower.push_back(c[j]);
      upper.push_back(c[j]);
      ++row;
    }
  }

  // Friction cone and force limits.
  const double a = config_.mu / std::sqrt(2.0);
  for (int k = 0; k < N; ++k) {
    for (int i = 0; i < kNumLegs; ++i) {
      const int base = u_index(k) + 3 * i;
      const std::array<std::array<double, 3>, 6> rows{{
          {1.0, 0.0, -a},
          {-1.0, 0.0, -a},
          {0.0, 1.0, -a},
          {0.0, -1.0, -a},
          {0.0, 0.0, -1.0},
          {0.0, 0.0, 1.0},
      }};
      for (int r = 0; r < 6; ++r) {
        for (int j = 0; j < 3; ++j) {
          if (rows[r][j] != 0.0) {
            a_triplets.emplace_back(row, base + j, rows[r][j]);
          }
        }
        lower.push_back(-kInf);
        upper.push_back(r == 5 ? config_.fz_max : 0.0);
        ++row;
      }
    }
  }

  // State bounds with slacks: p_z, roll, pitch.
  const std::array<int, 3> bounded_states{2, 6, 7};
  const std::array<double, 3> lower_bounds{
      config_.z_min, -config_.roll_pitch_max, -config_.roll_pitch_max};
  const std::array<double, 3> upper_bounds{
      config_.z_max, config_.roll_pitch_max, config_.roll_pitch_max};
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < ns; ++j) {
      const int state_idx = x_index(k) + bounded_states[j];
      const int slack_idx = s_index(k) + j;
      a_triplets.emplace_back(row, state_idx, 1.0);
      a_triplets.emplace_back(row, slack_idx, -1.0);
      lower.push_back(-kInf);
      upper.push_back(upper_bounds[j]);
      ++row;

      a_triplets.emplace_back(row, state_idx, -1.0);
      a_triplets.emplace_back(row, slack_idx, -1.0);
      lower.push_back(-kInf);
      upper.push_back(-lower_bounds[j]);
      ++row;

      a_triplets.emplace_back(row, slack_idx, 1.0);
      lower.push_back(0.0);
      upper.push_back(kInf);
      ++row;
    }
  }

  Eigen::SparseMatrix<double> A(row, n);
  A.setFromTriplets(a_triplets.begin(), a_triplets.end());

  CscOwner P_owner;
  CscOwner A_owner;
  P_owner.matrix = eigenToCsc(P);
  A_owner.matrix = eigenToCsc(A);

  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.eps_abs = 1e-4;
  settings.eps_rel = 1e-4;
  settings.max_iter = 2000;

  OSQPData data;
  data.n = static_cast<c_int>(n);
  data.m = static_cast<c_int>(row);
  data.P = P_owner.matrix;
  data.A = A_owner.matrix;
  data.q = q.data();
  data.l = lower.data();
  data.u = upper.data();

  OSQPWorkspace* workspace = nullptr;
  const c_int setup_status = osqp_setup(&workspace, &data, &settings);
  if (setup_status != 0 || workspace == nullptr) {
    output.status = static_cast<int>(setup_status);
    const auto end = std::chrono::steady_clock::now();
    output.solve_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return output;
  }

  osqp_solve(workspace);
  output.status = static_cast<int>(workspace->info->status_val);
  output.iterations = static_cast<int>(workspace->info->iter);
  output.cost = static_cast<double>(workspace->info->obj_val);
  output.success = workspace->info->status_val == OSQP_SOLVED;

  const c_float* solution = workspace->solution->x;
  for (int i = 0; i < kNumLegs; ++i) {
    const int base = u_index(0) + 3 * i;
    output.forces[i] =
        Vec3(solution[base], solution[base + 1], solution[base + 2]);
  }

  osqp_cleanup(workspace);
  const auto end = std::chrono::steady_clock::now();
  output.solve_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return output;
}

}  // namespace go2w
