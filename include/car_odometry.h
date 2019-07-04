#pragma once

#include <Eigen/Core>

namespace ghost {

/* CarOdometry
 *
 * Computes an odometry estimate (position, heading, velocity, and yaw rate) for a car like vehicle
 * based on wheel encoder pulses and average front wheel angle.
 *
 * Which encoders contribute to which state updates (translation/rotation) can be configured. Each
 * state variable can be updated with the front, rear, or both sets of encoders.
 *
 * The motion model is based off of Automotive Control Systems - For Engine, Driveline, and Vehicle,
 * Sections 8.3, 9.1, and 9.2.
 */
class CarOdometry {
  public:
    // Parameters for the CarOdometry class
    struct Parameters {
        // Pulse counts for one revolution of the front and rear wheels
        uint32_t N_front = 1;
        uint32_t N_rear = 1;

        // Maximum wheel angle [deg]
        double delta_max = 0.0;
        // Vehicle track width
        double b = 1.0;
        // Wheel diameter
        double d = 1.0;
        // Calibration coefficient for the track width (i.e. the track width used for calculations
        // is b * Cb)
        double Cb = 1.0;
        // Calibration coefficient for the wheel diameter (i.e. the wheel diameter used for
        // calculations is d * Cd)
        double Cd = 1.0;

        // Flags for when wheel encoders (front or rear) should be used for which state updates
        // (translation or rotation)
        bool use_front_for_ds_update = true;
        bool use_rear_for_ds_update = true;
        bool use_front_for_dpsi_update = true;
        bool use_rear_for_dpsi_update = true;
    };  // end Parameters

    // Vehicle (2D) state definition
    struct State {
        // Position vector
        Eigen::Vector2d position = Eigen::Vector2d::Zero();
        // Yaw angle
        double psi = 0.0;
        // Yaw rate
        double psi_dot = 0.0;
        // Velocity magnitude (always in the X direction)
        double velocity = 0.0;
    };  // end State

    // Count of pulses for each wheel
    struct Pulses {
        uint32_t FL;
        uint32_t FR;
        uint32_t RL;
        uint32_t RR;

        // Note, the difference between two pulse counts will always be positive
        Pulses operator-(const Pulses& other) const {
            Pulses result;
            result.FL = (FL >= other.FL) ? (FL - other.FL) : (other.FL - FL);
            result.FR = (FR >= other.FR) ? (FR - other.FR) : (other.FR - FR);
            result.RL = (RL >= other.RL) ? (RL - other.RL) : (other.RL - RL);
            result.RR = (RR >= other.RR) ? (RR - other.RR) : (other.RR - RR);

            return result;
        }
    };  // end Pulses

    /* Constructor
     *
     * @params: Parameters to use for odometry calculations
     */
    CarOdometry(const Parameters& params)
        : params_(params) {
        zeroState();
        processParams();
    }

    /* setState
     *
     * @param new_state: New value to assign to the current state
     */
    void setState(const State& new_state) { state_ = new_state; }

    /* getState
     *
     * @return: Current vehicle state
     */
    State getState() { return state_; }

    /* zeroState
     *
     * Resets all state variables to zero.
     */
    void zeroState() { state_ = State(); }

    /* updateState
     *
     * Updates the vehicle state based on the encoder pulses.
     *
     * The pulse counts are not absolute, they are relative to the counts since the previous update.
     *
     * @param pulses: Encoder pulse counts
     * @param steering: Steering input [-1, 1] (1=full left)
     * @param dt: Time duration the pulse counts cover
     */
    void updateState(const Pulses& pulses, double steering, double dt);

    /* pulsesToTranslation
     *
     * @param pulses: Encoder pulse counts
     * @param steering: Steering input [-1, 1] (1=full left)
     * @return: Linear distance travelled
     */
    double pulsesToTranslation(const Pulses& pulses, double steering) const;

    /* pulsesToRotation
     *
     * @param pulses: Encoder pulse counts
     * @param steering: Steering input [-1, 1] (1=full left)
     * @return: Change in heading angle [rad]
     */
    double pulsesToRotation(const Pulses& pulses, double steering) const;

    /* integrateMotion
     *
     * Computes a new state by integrating the motion over one step.
     *
     * This will use an exact intergation when possible, otherwise it will use a 2nd order Runge
     * Kutta method.
     *
     * @param init_state: Initial state to start integration from
     * @param dt: Time duration for integration
     * @param ds: Distance travelled
     * @param dpsi: Yaw angle change
     * @return: Resulting state from the motion integration
     */
    State integrateMotion(const State& init_state, double dt, double ds, double dpsi) const;

  protected:
    // Current vehicle state
    State state_;
    // Parameters for this object
    Parameters params_;

  private:
    /* steeringToWheelAngle
     *
     * This performs a linear interpolate based on the max wheel angle.
     *
     * @param steering: Steering input [-1, 1] (1=full left)
     * @return: Wheel angle for the steering input [rad]
     */
    double steeringToWheelAngle(double steering) const {
        steering = std::max(std::min(steering, 1.0), -1.0);
        return steering * delta_max_rad_;
    }

    /* processParams
     *
     * Pre computes some constants based on the parameter settings.
     */
    void processParams() {
        d_eff_ = params_.d * params_.Cd;
        b_eff_ = params_.b * params_.Cb;
        ds_per_rev_ = d_eff_ * M_PI;
        delta_max_rad_ = params_.delta_max * M_PI / 180.0;
    }

    // Effective wheel diameter
    double d_eff_;
    // Effective track width
    double b_eff_;
    // Distance travelled per revolution
    double ds_per_rev_;
    // Max steering angle in radians
    double delta_max_rad_;
};  // end CarOdometry

}  // namespace ghost
