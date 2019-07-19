#include "car_odometry_ros.h"
#include <Eigen/Geometry>
#include <boost/assign.hpp>

namespace ghost {

CarOdometryROS::CarOdometryROS(const ros::NodeHandle& nh,
                               const CarOdometryROS::Parameters& node_params)
    : nh_(nh)
    , params_(node_params)
    , first_update_performed_(false) {
    // Setup the odometry object
    odometry_.reset(new CarOdometry(params_.odom_params));

    // Initialize our message frame data
    odom_msg_.header.frame_id = params_.world_frame;
    odom_msg_.child_frame_id = params_.base_frame;

    tf_msg_.header.frame_id = params_.world_frame;
    tf_msg_.child_frame_id = params_.base_frame;

    // Initialize the odometry covariance matrices
    // clang-format off
    std::vector<double> cov;
    cov = params_.pose_covariance_diagonal;
    ROS_ASSERT(cov.size() == 6);
    odom_msg_.pose.covariance = boost::assign::list_of
        (cov[0]) (0)  (0)  (0)  (0)  (0)
        (0)  (cov[1]) (0)  (0)  (0)  (0)
        (0)  (0)  (cov[2]) (0)  (0)  (0)
        (0)  (0)  (0)  (cov[3]) (0)  (0)
        (0)  (0)  (0)  (0)  (cov[4]) (0)
        (0)  (0)  (0)  (0)  (0)  (cov[5]);

    cov = params_.twist_covariance_diagonal;
    ROS_ASSERT(cov.size() == 6);
    odom_msg_.twist.covariance = boost::assign::list_of
        (cov[0]) (0)  (0)  (0)  (0)  (0)
        (0)  (cov[1]) (0)  (0)  (0)  (0)
        (0)  (0)  (cov[2]) (0)  (0)  (0)
        (0)  (0)  (0)  (cov[3]) (0)  (0)
        (0)  (0)  (0)  (0)  (cov[4]) (0)
        (0)  (0)  (0)  (0)  (0)  (cov[5]);
    // clang-format on

    // Setup the publisher
    pub_odom_ = nh_.advertise<nav_msgs::Odometry>(params_.odom_topic, 50);

    // Setup the transform broadcaster
    if (params_.broadcast_tf) {
        tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster());
    }

    ROS_INFO("Car odometry initialized");
}

void CarOdometryROS::update(const CarOdometry::Pulses& pulses, double steering,
                            const ros::Time& time) {
    if (first_update_performed_) {
        const double dt = (time - prev_update_time_).toSec();
        prev_update_time_ = time;

        const CarOdometry::Pulses rel_pulses = pulses - prev_pulses_;
        prev_pulses_ = pulses;

        odometry_->updateState(rel_pulses, steering, dt);
    } else {
        // Need to record this input and wait for the next update to actually compute the change in
        // the state
        prev_update_time_ = time;
        prev_pulses_ = pulses;
        first_update_performed_ = true;
    }
}

void CarOdometryROS::broadcast() {
    publishOdometry();
    if (params_.broadcast_tf) {
        publishTF();
    }
}

void CarOdometryROS::publishOdometry() {
    const CarOdometry::State& state = odometry_->getState();

    odom_msg_.header.stamp = prev_update_time_;
    odom_msg_.header.stamp = prev_update_time_;
    odom_msg_.pose.pose.position.x = state.position.x();
    odom_msg_.pose.pose.position.y = state.position.y();
    odom_msg_.twist.twist.linear.x = state.velocity;
    odom_msg_.twist.twist.angular.z = state.psi_dot;

    Eigen::Quaterniond quat =
        Eigen::Quaterniond(Eigen::AngleAxisd(state.psi, Eigen::Vector3d::UnitZ()));
    odom_msg_.pose.pose.orientation.x = quat.x();
    odom_msg_.pose.pose.orientation.y = quat.y();
    odom_msg_.pose.pose.orientation.z = quat.z();
    odom_msg_.pose.pose.orientation.w = quat.w();

    pub_odom_.publish(odom_msg_);
}

void CarOdometryROS::publishTF() {
    const CarOdometry::State& state = odometry_->getState();

    tf_msg_.header.stamp = prev_update_time_;
    tf_msg_.transform.translation.x = state.position.x();
    tf_msg_.transform.translation.y = state.position.y();

    Eigen::Quaterniond quat =
        Eigen::Quaterniond(Eigen::AngleAxisd(state.psi, Eigen::Vector3d::UnitZ()));
    tf_msg_.transform.rotation.x = quat.x();
    tf_msg_.transform.rotation.y = quat.y();
    tf_msg_.transform.rotation.z = quat.z();
    tf_msg_.transform.rotation.w = quat.w();

    tf_broadcaster_->sendTransform(tf_msg_);
}

}  // namespace ghost
