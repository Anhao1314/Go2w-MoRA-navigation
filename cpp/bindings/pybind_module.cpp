#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "go2w/controller.hpp"

namespace py = pybind11;

namespace {

using DoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

void requireSize(const DoubleArray& array, py::ssize_t expected,
                 const char* name) {
  if (array.ndim() != 1 || array.shape(0) != expected) {
    throw std::invalid_argument(std::string(name) + " must have length " +
                                std::to_string(expected));
  }
}

class PyS0Controller {
 public:
  explicit PyS0Controller(bool pre_brake = false)
      : controller_(makeConfig(pre_brake)) {}

  void reset() { controller_.reset(go2w::RobotState{}); }

  DoubleArray stepLegacy(const DoubleArray& base_p,
                         const DoubleArray& base_q_xyzw,
                         const DoubleArray& q_leg,
                         const DoubleArray& q_wheel, double speed_scale,
                         double turn_adjust, int segment_idx,
                         double dist_to_subgoal, double speed) {
    requireSize(base_p, 3, "base_p");
    requireSize(base_q_xyzw, 4, "base_q_xyzw");
    requireSize(q_leg, 12, "q_leg");
    requireSize(q_wheel, 4, "q_wheel");

    go2w::RobotState state;
    const auto p = base_p.unchecked<1>();
    const auto q = base_q_xyzw.unchecked<1>();
    const auto ql = q_leg.unchecked<1>();
    const auto qw = q_wheel.unchecked<1>();
    state.base.p = go2w::Vec3(p(0), p(1), p(2));
    state.base.q = Eigen::Quaterniond(q(3), q(0), q(1), q(2));
    for (int i = 0; i < 12; ++i) {
      state.joints.q_leg[i] = ql(i);
    }
    for (int i = 0; i < 4; ++i) {
      state.joints.q_wheel[i] = qw(i);
    }

    last_ = controller_.stepLegacy(state, speed_scale, turn_adjust,
                                   segment_idx, dist_to_subgoal, speed);
    DoubleArray result(6);
    auto out = result.mutable_unchecked<1>();
    for (int i = 0; i < 6; ++i) {
      out(i) = last_.action6[i];
    }
    return result;
  }

  double totalMass() const { return controller_.model().totalMass(); }
  int lastSafety() const { return static_cast<int>(last_.safety); }
  bool lastValid() const { return last_.valid; }

 private:
  static go2w::S0Config makeConfig(bool pre_brake) {
    go2w::S0Config config;
    config.path.pre_brake = pre_brake;
    return config;
  }

  go2w::S0Controller controller_;
  go2w::ControlOutput last_;
};

}  // namespace

PYBIND11_MODULE(go2w_py, m) {
  m.doc() = "Go2W C++ System-0 controller (P0 portfolio build)";
  py::class_<PyS0Controller>(m, "S0Controller")
      .def(py::init<bool>(), py::arg("pre_brake") = false)
      .def("reset", &PyS0Controller::reset)
      .def("step_legacy", &PyS0Controller::stepLegacy, py::arg("base_p"),
           py::arg("base_q_xyzw"), py::arg("q_leg"), py::arg("q_wheel"),
           py::arg("speed_scale"), py::arg("turn_adjust"),
           py::arg("segment_idx"), py::arg("dist_to_subgoal"),
           py::arg("speed"))
      .def("total_mass", &PyS0Controller::totalMass)
      .def("last_safety", &PyS0Controller::lastSafety)
      .def("last_valid", &PyS0Controller::lastValid);
}
