#pragma once

#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include "car_odometry.h"

namespace ghost {

/* CarOdometryROS
 *
 * ROS wrapper around the CarOdometry class that will publish odometry and transform messages with
 * state updates.
 */
class CarOdometryROS {
  public:
    // Parameters for ROS node settings
    struct Parameters {
        // Topic to publish odometry messages on
        std::string odom_topic = "odometry";
        // Vehicle frame the odometry estimate is computed for
        std::string base_frame = "base_link";
        // World frame the odometry pose is expressed in
        std::string world_frame = "odom";
        // Flag for if the transform should be published
        bool broadcast_tf = true;
        // Covariance to assign to the diagonal entries in the pose covariance matrix
        std::vector<double> pose_covariance_diagonal = std::vector<double>(6, 1e-3);
        // Covariance to assign to the diagonal entries in the twist covariance matrix
        std::vector<double> twist_covariance_diagonal = std::vector<double>(6, 1e-3);
        // Parameters for the odometry calculations
        CarOdometry::Parameters odom_params;
    };

    /* Constructor
     *
     * @param nh: Nodehandle to publish with
     * @node_params: Parameters for
     */
    CarOdometryROS(const ros::NodeHandle& nh, const CarOdometryROS::Parameters& node_params);

    /* update
     *
     * Updates the odometry estimate and publishes odometry and TF updates.
     *
     * @param pulses: Absolute pulse counts for each wheel
     * @param steering: Steering input [-1, 1] (1=full left)
     * @param time: Time of the encoder data, this is the timestamp that will be set on the
     * published data
     */
    void update(const CarOdometry::Pulses& pulses, double steering, const ros::Time& time);

  protected:
    /* publishOdometry
     *
     * Publishes the current odometry state as an Odometry message.
     *
     * @param time: Time to stamp the message with
     */
    void publishOdometry(const ros::Time& time);

    /* broadcastTF
     *
     * Publishes a transform between the world frame and the base frame for the current odometry
     * state.
     *
     * @param time: Time to stamp the transform with
     */
    void broadcastTF(const ros::Time& time);

    // Parameters for this node
    Parameters params_;
    // Odometry estimator
    std::unique_ptr<CarOdometry> odometry_;
    // Nodehandle to publish with
    ros::NodeHandle nh_;
    // Pubslisher of odometry message
    ros::Publisher pub_odom_;
    // TF broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    // Odometry message to update and publish (has pre set data in it)
    nav_msgs::Odometry odom_msg_;
    // Transform to broadcast (has pre set data in it)
    geometry_msgs::TransformStamped tf_msg_;
    // Time of the previous update, for computing the time interval
    ros::Time prev_update_time_;
    // Previous pulse counts, for computing the relative pulse count changes
    CarOdometry::Pulses prev_pulses_;
    // Flag for if the first update has been performed or not, we need at least two updates in
    // order to compute the pulse difference
    bool first_update_performed_;
};  // end CarOdometryROS

}  // namespace ghost
