#include <ghost/ControlState.h>
#include <ros/ros.h>
#include "car_odometry_ros.h"

using namespace ghost;

std::unique_ptr<CarOdometryROS> odom;

/* parseCovarianceParameter
 *
 * Parses a parameter list of covarience values as strings and converts them to a floating point
 * format will handle scientific notation and lack of decimals.
 *
 * @param val_strings: Strings to parse
 * @return: Floating point representation of parameter values.
 */
std::vector<double> parseCovarianceParameter(const std::vector<std::string>& val_strings) {
    std::vector<double> cov_vals(6);

    ROS_ASSERT(val_strings.size() == 6);
    for (int i = 0; i < val_strings.size(); ++i) {
        std::istringstream istr(val_strings[i]);
        istr >> cov_vals[i];
    }

    return cov_vals;
}

void controlStateCallback(const ghost::ControlState::ConstPtr& msg) {
    // Feed the data into the odometry estimator
    CarOdometry::Pulses pulses = {
        static_cast<int>(msg->FL_pulse_count), static_cast<int>(msg->FR_pulse_count),
        static_cast<int>(msg->RL_pulse_count), static_cast<int>(msg->RR_pulse_count)};

    odom->update(pulses, msg->car_control.steering, msg->header.stamp);
    odom->broadcast();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "odometry_integration");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // Load all the parameters
    std::string control_state_topic;
    pnh.param<std::string>("control_state_topic", control_state_topic, "control_state");

    CarOdometryROS::Parameters params;
    pnh.getParam("/chassis/delta_max", params.odom_params.delta_max);
    pnh.getParam("/chassis/b", params.odom_params.b);
    pnh.getParam("/chassis/d", params.odom_params.d);
    pnh.getParam("/chassis/Cb", params.odom_params.Cb);
    pnh.getParam("/chassis/Cd", params.odom_params.Cd);
    pnh.getParam("odom_topic", params.odom_topic);
    pnh.getParam("base_frame", params.base_frame);
    pnh.getParam("world_frame", params.world_frame);
    pnh.getParam("broadcast_tf", params.broadcast_tf);
    int N_front, N_rear;
    if (pnh.hasParam("N_front")) {
        pnh.getParam("N_front", N_front);
        params.odom_params.N_front = static_cast<uint32_t>(N_front);
    }
    if (pnh.hasParam("N_rear")) {
        pnh.getParam("N_rear", N_rear);
        params.odom_params.N_rear = static_cast<uint32_t>(N_rear);
    }
    pnh.getParam("use_front_for_ds_update", params.odom_params.use_front_for_ds_update);
    pnh.getParam("use_rear_for_ds_update", params.odom_params.use_rear_for_ds_update);
    pnh.getParam("use_front_for_dpsi_update", params.odom_params.use_front_for_dpsi_update);
    pnh.getParam("use_rear_for_dpsi_update", params.odom_params.use_rear_for_dpsi_update);
    if (pnh.hasParam("pose_covariance_diagonal")) {
        std::vector<std::string> pose_cov_list;
        pnh.getParam("pose_covariance_diagonal", pose_cov_list);
        params.pose_covariance_diagonal = parseCovarianceParameter(pose_cov_list);
    }
    if (pnh.hasParam("twist_covariance_diagonal")) {
        std::vector<std::string> twist_cov_list;
        pnh.getParam("twist_covariance_diagonal", twist_cov_list);
        params.twist_covariance_diagonal = parseCovarianceParameter(twist_cov_list);
    }

    // Initialize our odometry estimator
    odom.reset(new CarOdometryROS(nh, params));

    // Setup the subscriber for messages, and spin
    ros::Subscriber state_sub = nh.subscribe(control_state_topic, 50, controlStateCallback);
    ros::spin();
}
