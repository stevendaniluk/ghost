#include <gtest/gtest.h>
#include "car_odometry.h"

using namespace ghost;

class CarOdometryTest : public ::testing::Test {
  public:
    virtual void SetUp() {
        // Set some default parameters
        params.N_front = 10;
        params.N_rear = 10;
        params.b = 0.50;
        params.d = 0.20;
        params.delta_max = 30 * M_PI / 180.0;
        params.Cb = 1.0;
        params.Cd = 1.0;
        params.use_front_for_ds_update = true;
        params.use_rear_for_ds_update = true;
        params.use_front_for_dpsi_update = true;
        params.use_rear_for_dpsi_update = true;

        // Initialize the odometry
        odometry.reset(new CarOdometry(params));
    }

    std::unique_ptr<CarOdometry> odometry;
    CarOdometry::Parameters params;
};

TEST_F(CarOdometryTest, StateInitializedToZero) {
    CarOdometry::State state = odometry->getState();
    EXPECT_EQ(0, state.position.x());
    EXPECT_EQ(0, state.position.y());
    EXPECT_EQ(0, state.psi);
    EXPECT_EQ(0, state.psi_dot);
    EXPECT_EQ(0, state.velocity);
}

TEST_F(CarOdometryTest, EqualPulsesTranslationIsCorrect) {
    int n = 15;
    double ds_per_rev = params.d * params.Cd * M_PI;
    double ds_expected = ds_per_rev * n / params.N_front;

    CarOdometry::Pulses pulses = {n, n, n, n};
    double ds = odometry->pulsesToTranslation(pulses, 0);
    EXPECT_FLOAT_EQ(ds_expected, ds);
}

TEST_F(CarOdometryTest, TranslationScalesWithWheelAngle) {
    // Only use the front wheels for integration
    params.use_rear_for_ds_update = false;
    odometry.reset(new CarOdometry(params));

    // Get a baseline with zero wheel angle
    int n = 15;
    CarOdometry::Pulses pulses = {n, n, n, n};
    double ds_straight_wheel = odometry->pulsesToTranslation(pulses, 0);

    // The amount we move forward should be the measured distance projected onto the vehicle
    // direction
    double steering = 1.0;
    double delta = params.delta_max * M_PI / 180.0;
    double ds_expected = cos(delta) * ds_straight_wheel;
    double ds = odometry->pulsesToTranslation(pulses, steering);

    EXPECT_FLOAT_EQ(ds_expected, ds);
}

TEST_F(CarOdometryTest, TranslationIsMeanOfAllWheels) {
    int n_low = 10;
    int n_high = 20;
    int n_avg = (n_low + n_high) / 2;

    // Put the high and low pulse counts on opposite corners
    CarOdometry::Pulses pulses_unequal = {n_low, n_high, n_high, n_low};
    double ds = odometry->pulsesToTranslation(pulses_unequal, 0);

    // Actual translation should be the same as with the average pulses over all wheels
    CarOdometry::Pulses pulses_equal = {n_avg, n_avg, n_avg, n_avg};
    double ds_expected = odometry->pulsesToTranslation(pulses_equal, 0);

    EXPECT_FLOAT_EQ(ds_expected, ds);
}

TEST_F(CarOdometryTest, EqualPulsesZeroRotation) {
    int n = 15;
    CarOdometry::Pulses pulses = {n, n, n, n};
    double dpsi = odometry->pulsesToRotation(pulses, 0);
    EXPECT_FLOAT_EQ(0.0, dpsi);
}

TEST_F(CarOdometryTest, HalfCircleRotationIsCorrect) {
    // Only use the rear wheels for integration
    params.use_front_for_dpsi_update = false;
    odometry.reset(new CarOdometry(params));

    // Determine the number of pulses we'd expect when driving half way around a circle
    double b_eff = params.b * params.Cb;
    double ds_per_rev = params.d * params.Cd * M_PI;

    double r = 3.0;
    double r_inner = r - 0.5 * b_eff;
    double r_outer = r + 0.5 * b_eff;

    double ds_inner = M_PI * r_inner;
    double ds_outer = M_PI * r_outer;

    int n_inner = params.N_rear * ds_inner / ds_per_rev;
    int n_outer = params.N_rear * ds_outer / ds_per_rev;

    CarOdometry::Pulses pulses = {n_inner, n_outer, n_inner, n_outer};

    double dpsi_expected = M_PI;
    double dpsi = odometry->pulsesToRotation(pulses, 0);
    EXPECT_NEAR(dpsi_expected, dpsi, 0.1);
}

TEST_F(CarOdometryTest, RotationIsMeanOfFrontAndRear) {
    int n_inner = 10;
    int n_outer = 20;

    // Compute the rotation rate for a baseline
    CarOdometry::Pulses pulses = {n_inner, n_outer, n_inner, n_outer};
    double dpsi = odometry->pulsesToRotation(pulses, 0);

    // Now increase the wheel rate difference on the front, and lower it in the rear, it should
    // result in the same net rotation
    CarOdometry::Pulses pulses_perturbed = {n_inner - 2, n_outer + 2, n_inner + 2, n_outer - 2};
    double dpsi_perturbed = odometry->pulsesToRotation(pulses_perturbed, 0);

    EXPECT_FLOAT_EQ(dpsi, dpsi_perturbed);
}

TEST_F(CarOdometryTest, DisableInputsForTranslation) {
    int n = 15;
    CarOdometry::Pulses pulses_front_only = {n, n, 0, 0};
    CarOdometry::Pulses pulses_rear_only = {0, 0, n, n};

    // Try with front disabled and only front wheel rotation
    params.use_front_for_ds_update = false;
    params.use_rear_for_ds_update = true;
    odometry.reset(new CarOdometry(params));

    double ds_front_disabled = odometry->pulsesToTranslation(pulses_front_only, 0);
    EXPECT_FLOAT_EQ(0, ds_front_disabled);

    // Try with rear disabled and only rear wheel rotation
    params.use_front_for_ds_update = true;
    params.use_rear_for_ds_update = false;
    odometry.reset(new CarOdometry(params));

    double ds_rear_disabled = odometry->pulsesToTranslation(pulses_rear_only, 0);
    EXPECT_FLOAT_EQ(0, ds_front_disabled);
}

TEST_F(CarOdometryTest, DisableInputsForRotation) {
    // Make some pulses to make the vehicle turn left
    int n = 15;
    CarOdometry::Pulses pulses_front_only = {n, 2 * n, 0, 0};
    CarOdometry::Pulses pulses_rear_only = {0, 0, n, 2 * n};

    // Try with front disabled and only front wheel rotation
    params.use_front_for_dpsi_update = false;
    params.use_rear_for_dpsi_update = true;
    odometry.reset(new CarOdometry(params));

    double dpsi_front_disabled = odometry->pulsesToRotation(pulses_front_only, 0);
    EXPECT_FLOAT_EQ(0, dpsi_front_disabled);

    // Try with rear disabled and only rear wheel rotation
    params.use_front_for_dpsi_update = true;
    params.use_rear_for_dpsi_update = false;
    odometry.reset(new CarOdometry(params));

    double dpsi_rear_disabled = odometry->pulsesToRotation(pulses_rear_only, 0);
    EXPECT_FLOAT_EQ(0, dpsi_rear_disabled);
}

TEST_F(CarOdometryTest, MotionIntegrationPureTranslation) {
    CarOdometry::State state;
    double dt = 0.5;
    double ds = 1.2;
    double dpsi = 0.0;
    state = odometry->integrateMotion(state, dt, ds, dpsi);
    EXPECT_FLOAT_EQ(ds, state.position.x());
    EXPECT_FLOAT_EQ(0, state.position.y());
}

TEST_F(CarOdometryTest, MotionIntegrationPureRotation) {
    CarOdometry::State state;
    double dt = 0.5;
    double ds = 0.0;
    double dpsi = 1.2;
    state = odometry->integrateMotion(state, dt, ds, dpsi);
    EXPECT_FLOAT_EQ(dpsi, state.psi);
}

TEST_F(CarOdometryTest, MotionIntegrationHalfCircle) {
    // Assume a heading an arc length for moving 1/4 way around a constant radius circle
    double r = 3.0;
    double dpsi = M_PI / 2;
    double ds = dpsi * r;

    CarOdometry::State state;
    double dt = 0.5;
    state = odometry->integrateMotion(state, dt, ds, dpsi);

    EXPECT_NEAR(r, state.position.x(), 1e-6);
    EXPECT_NEAR(r, state.position.y(), 1e-6);
    EXPECT_NEAR(dpsi, state.psi, 1e-6);
}

TEST_F(CarOdometryTest, VelocityUpdate) {
    CarOdometry::State state;
    double dt = 0.5;
    double ds = 0.0;
    double dpsi = 1.2;
    state = odometry->integrateMotion(state, dt, ds, dpsi);

    double v_expected = ds / dt;
    EXPECT_FLOAT_EQ(v_expected, state.velocity);
}

TEST_F(CarOdometryTest, YawRateUpdate) {
    CarOdometry::State state;
    double dt = 0.5;
    double ds = 0.0;
    double dpsi = 1.2;
    state = odometry->integrateMotion(state, dt, ds, dpsi);

    double psi_dot_expected = dpsi / dt;
    EXPECT_FLOAT_EQ(psi_dot_expected, state.psi_dot);
}
