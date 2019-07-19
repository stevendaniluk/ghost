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
    double revs_F = 0;
    double revs_R = 0;
    if (params_.use_front_for_ds_update) {
        const double revs = (pulses.FL + pulses.FR) / (2.0 * params_.N_front);
        const double delta = steeringToWheelAngle(steering);
        revs_F = revs * cos(delta);
    }
    if (params_.use_rear_for_ds_update) {
        revs_R = (pulses.RL + pulses.RR) / (2.0 * params_.N_rear);
    }

    const double ds = ds_per_rev_ * (revs_F + revs_R) / num_inputs;
    return ds;
}

double CarOdometry::pulsesToRotation(const Pulses& pulses, double steering) const {
    uint32_t num_inputs = params_.use_front_for_dpsi_update + params_.use_rear_for_dpsi_update;
    if (num_inputs == 0) {
        return 0.0;
    }

    // Rotation is the average of all four wheels
    double rev_diff_F = 0;
    double rev_diff_R = 0;
    if (params_.use_front_for_dpsi_update) {
        const double rev_diff = static_cast<double>(pulses.FR - pulses.FL) / params_.N_front;
        const double delta = steeringToWheelAngle(steering);
        rev_diff_F = rev_diff / cos(delta);
    }
    if (params_.use_rear_for_dpsi_update) {
        rev_diff_R = static_cast<double>(pulses.RR - pulses.RL) / params_.N_front;
    }

    const double dpsi = ds_per_rev_ * (rev_diff_F + rev_diff_R) / (b_eff_ * num_inputs);
    return dpsi;
}

CarOdometry::State CarOdometry::integrateMotion(const State& init_state, double dt, double ds,
                                                double dpsi) const {
    if (dt == 0) {
        return init_state;
    }

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
