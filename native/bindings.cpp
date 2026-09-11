// Simulation-only adapter. The controller sources remain in their own checkout.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include "Digit_Controller.hpp"
namespace py = pybind11;
using NumpyArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void check(const NumpyArray& a, std::initializer_list<py::ssize_t> shape) {
    if (a.ndim() != static_cast<py::ssize_t>(shape.size()) ||
        !std::equal(shape.begin(), shape.end(), a.shape()))
        throw py::value_error("Unexpected observation/limits array shape");
    for (py::ssize_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a.data()[i])) throw py::value_error("NumpyArrays must be finite");
}

class Controller {
    Digit_Controller controller;
    llapi_limits_t limits{};
    double last_time = -1;
public:
    Controller(int mode, const NumpyArray& bounds, double period) {
        if (mode < 0 || mode > 2) throw py::value_error("Mode must be 0, 1, or 2");
        if (!std::isfinite(period) || period <= 0) throw py::value_error("Invalid control period");
        check(bounds, {3, NUM_MOTORS});
        for (py::ssize_t i = 0; i < bounds.size(); ++i)
            if (bounds.data()[i] <= 0) throw py::value_error("Limits must be positive");
        std::copy_n(bounds.data(), NUM_MOTORS, limits.torque_limit);
        std::copy_n(bounds.data()+NUM_MOTORS, NUM_MOTORS, limits.damping_limit);
        std::copy_n(bounds.data()+2*NUM_MOTORS, NUM_MOTORS, limits.velocity_limit);
        controller.Initialize_(mode, 1);
        controller.time_digit_prev_ = -1;
        controller.command_update_prev_ = {};
        controller.kf_sample_time_ = period;
        controller.Set_Initial_Standing_Gains_();
        controller.Set_Initial_Walking_Gains_();
    }
    void set_mode(int mode) {
        if (mode < 0 || mode > 2) throw py::value_error("Mode must be 0, 1, or 2");
        if (mode != controller.ctrl_mode_) {
            if (mode == 2) controller.flag_walking_first_iter_ = true;
            else controller.flag_standing_first_iter_ = true;
        }
        controller.Set_Ctrl_Mode_(mode);
    }
    void set_velocity(double forward, double lateral, double turn) {
        if (!std::isfinite(forward) || !std::isfinite(lateral) || !std::isfinite(turn))
            throw py::value_error("Velocity targets must be finite");
        controller.vel_x_des_tuned_ = forward;
        controller.vel_y_des_tuned_ = lateral;
        controller.turn_rps_tuned_ = turn;
    }
    NumpyArray update(double time, const NumpyArray& base, const NumpyArray& motors, const NumpyArray& joints) {
        if (!std::isfinite(time) || time < 0 || time <= last_time)
            throw py::value_error("Controller time must be nonnegative and strictly increasing; recreate on reset");
        check(base, {13}); check(motors, {3, NUM_MOTORS}); check(joints, {2, NUM_UNACT_JOINTS});
        llapi_observation_t obs{};
        obs.time = time;
        std::copy_n(base.data(), 3, obs.base.translation);
        const auto* b = base.data();
        double norm = std::sqrt(b[3]*b[3]+b[4]*b[4]+b[5]*b[5]+b[6]*b[6]);
        if (norm < 1e-12) throw py::value_error("Invalid base quaternion");
        obs.base.orientation = {b[3]/norm,b[4]/norm,b[5]/norm,b[6]/norm};
        obs.imu.orientation = obs.base.orientation;
        std::copy_n(b+7, 3, obs.base.linear_velocity);
        std::copy_n(b+10, 3, obs.base.angular_velocity);
        std::copy_n(motors.data(), NUM_MOTORS, obs.motor.position);
        std::copy_n(motors.data()+NUM_MOTORS, NUM_MOTORS, obs.motor.velocity);
        std::copy_n(motors.data()+2*NUM_MOTORS, NUM_MOTORS, obs.motor.torque);
        std::copy_n(joints.data(), NUM_UNACT_JOINTS, obs.joint.position);
        std::copy_n(joints.data()+NUM_UNACT_JOINTS, NUM_UNACT_JOINTS, obs.joint.velocity);
        llapi_command_t command{};
        controller.Update_(command, obs, &limits);
        last_time = time;
        NumpyArray output({NUM_MOTORS, 3});
        auto out = output.mutable_unchecked<2>();
        for (int i=0; i<NUM_MOTORS; ++i) {
            out(i,0)=command.motors[i].torque;
            out(i,1)=command.motors[i].velocity;
            out(i,2)=command.motors[i].damping;
        }
        return output; // owns its memory; no views into controller internals
    }
};
PYBIND11_MODULE(_digit_alip, m) {
    py::class_<Controller>(m, "Controller")
        .def(py::init<int, const NumpyArray&, double>(), py::arg("mode"), py::arg("limits"), py::arg("period"))
        .def("set_mode", &Controller::set_mode)
        .def("set_velocity", &Controller::set_velocity)
        .def("update", &Controller::update, py::arg("time"), py::arg("base"), py::arg("motors"), py::arg("joints"));
}
