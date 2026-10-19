#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "go2w/types.hpp"

namespace go2w::test {

inline std::vector<std::vector<double>> readNumericCsv(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot open fixture: " + path);
  }
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream stream(line);
    std::vector<double> row;
    double value = 0.0;
    while (stream >> value) {
      row.push_back(value);
    }
    if (!row.empty()) {
      rows.push_back(std::move(row));
    }
  }
  return rows;
}

// State row layout:
//   base_p(3) base_q_xyzw(4) q_leg(12) q_wheel(4)
inline RobotState stateFromRow(const std::vector<double>& row) {
  if (row.size() < 23) {
    throw std::runtime_error("state row too short");
  }
  RobotState state;
  state.base.p = Vec3(row[0], row[1], row[2]);
  state.base.q = Eigen::Quaterniond(row[6], row[3], row[4], row[5]);
  for (int i = 0; i < 12; ++i) {
    state.joints.q_leg[i] = row[7 + i];
  }
  for (int i = 0; i < 4; ++i) {
    state.joints.q_wheel[i] = row[19 + i];
  }
  return state;
}

}  // namespace go2w::test
