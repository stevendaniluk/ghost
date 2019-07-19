#include "car_odometry.h"

namespace ghost {

void CarOdometry::updateState(const Pulses& pulses, double steering, double dt) {
    const double ds = pulsesToTranslation(pulses, steering);
    const double dpsi = pulsesToRotation(pulses, steering);
    state_ = integrateMotion(state_, dt, ds, dpsi);
}

double CarOdometry::pulsesToTranslation(const Pulses& pulses, double steering) const {
    uint32_t num_inputs = params_.use_front_for_ds_update + params_.use_rear_for_ds_update;
    if (num_inputs == 0) {
        return 0.0;
    }

    // Translation is the average of all four wheels
    double ds_F = 0;
    double ds_R = 0;
    if (params_.use_front_for_ds_update) {
        const double revs = (pulses.FL + pulses.FR) / (2.0 * params_.N_front);
        const double delta = steeringToWheelAngle(steering);
        ds_F = ds_per_rev_ * revs * cos(delta);
    }
    if (params_.use_rear_for_ds_update) {
        const double revs = (pulses.RL + pulses.RR) / (2.0 * params_.N_rear);
        ds_R = ds_per_rev_ * revs;
    }

    const double ds = (ds_F + ds_R) / num_inputs;
    return ds;
}

double CarOdometry::pulsesToRotation(const Pulses& pulses, double steering) const {
    uint32_t num_inputs = params_.use_front_for_dpsi_update + params_.use_rear_for_dpsi_update;
    if (num_inputs == 0) {
        return 0.0;
    }

    // Rotation is the average of all four wheels
    double dpsi_F = 0;
    double dpsi_R = 0;
    if (params_.use_front_for_dpsi_update) {
        const double rev_diff = static_cast<double>(pulses.FR - pulses.FL) / params_.N_front;
        const double delta = steeringToWheelAngle(steering);
        dpsi_F = ds_per_rev_ * rev_diff / (b_eff_ * cos(delta));
    }
    if (params_.use_rear_for_dpsi_update) {
        const double rev_diff = static_cast<double>(pulses.RR - pulses.RL) / params_.N_front;
        dpsi_R = ds_per_rev_ * rev_diff / b_eff_;
    }

    const double dpsi = (dpsi_F + dpsi_R) / num_inputs;
    return dpsi;
}

CarOdometry::State CarOdometry::integrateMotion(const State& init_state, double dt, double ds,
                                                double dpsi) const {
    State new_state;

    // Update the state rates
    new_state.velocity = ds / dt;
    new_state.psi_dot = dpsi / dt;

    // Integrate the motion
    new_state.position = init_state.position;
    if (fabs(new_state.psi_dot) > 1e-6) {
        // Perform exact integration
        const double r = ds / dpsi;
        const double new_psi = init_state.psi + dpsi;
        new_state.position.x() += r * (sin(new_psi) - sin(init_state.psi));
        new_state.position.y() -= r * (cos(new_psi) - cos(init_state.psi));
        new_state.psi = new_psi;
    } else {
        // Perform 2nd order Runge-Kutta integration
        new_state.position.x() += ds * cos(init_state.psi + 0.5 * dpsi);
        new_state.position.y() += ds * sin(init_state.psi + 0.5 * dpsi);
        new_state.psi = init_state.psi + dpsi;
    }

    return new_state;
}

}  // namespace ghost
