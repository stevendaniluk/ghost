/* cmd_car_to_twist
*
* Converts CarControl messages to their corresponding Twist messages, based
* on the car's maximum velocity, maximum steering angle, and acceleration
* limits (forward and braking).
*
* --------------------------------------------------------------------------
*/

#include <ros/ros.h>
#include <ghost/CarControl.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>

#define PI 3.14159265

class Converter {
  public:
    ros::NodeHandle nh_;
    ros::Publisher twist_pub_;  // Outgoing Twist messages
    ros::Subscriber pose_sub_;  // Car pose
    ros::Subscriber cmd_sub_;   // CarControl commands
    
    // Car parameters
    double l_;               // Car wheelbase
    double delta_max_;       // Maximum steering angle
    double v_max_;           // Maximum velocity
    double ax_max_accel_;    // Maximum longitudinal acceleration forward
    double ax_max_braking_;  // Maximum longitudinal acceleration braking
    
    // Misc.
    double v_;                  // Current car velocity
    ros::Time last_pose_time_;  // Time of last pose message received
    ros::Time last_cmd_time_;   // Time of last command message received
    bool first_cmd_;            // Flag for if this is the first command message
    
    Converter();
    void poseCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void cmdCarCallback(const ghost::CarControl::ConstPtr& msg);
};

/* Converter
*
* Constructor
*/
Converter::Converter() {
  // Need some chassis parameters, possible in another namespace
  std::string key;
  if(nh_.searchParam("chassis/l", key))
    nh_.getParam(key, l_);
  else
    l_ = 0.260;
  
  if(nh_.searchParam("chassis/delta_max", key))
    nh_.getParam(key, delta_max_);
  else
    delta_max_ = 25.0;
  delta_max_ *= PI/180;
  
  if(nh_.searchParam("chassis/v_max", key))
    nh_.getParam(key, v_max_);
  else
    v_max_ = 15.0;
  
  if(nh_.searchParam("chassis/ax_max_accel", key))
    nh_.getParam(key, ax_max_accel_);
  else
    ax_max_accel_ = 10.0;
  
  if(nh_.searchParam("chassis/ax_max_braking", key))
    nh_.getParam(key, ax_max_braking_);
  else
    ax_max_braking_ = 1.0;
  
  first_cmd_ = true;
  
  // Setup pubs and subs
  twist_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_car_twist", 1, true);
  pose_sub_ = nh_.subscribe("odometry/filtered", 50, &Converter::poseCallback, this);
  cmd_sub_ = nh_.subscribe("cmd_car", 50, &Converter::cmdCarCallback, this);
}

/* poseCallback
*
* Saves the car's state.
*/
void Converter::poseCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  // Save new state
  v_ = sqrt(pow(msg->twist.twist.linear.x, 2) + pow(msg->twist.twist.linear.y, 2));
  v_ = v_*(msg->twist.twist.linear.x >= 0) - v_*(msg->twist.twist.linear.x < 0);
  last_pose_time_ = msg->header.stamp;
}

/* cmdCarCallback
*
* Converts incoming CarControl messages to twist messages based on the
* car velocity and acceleration limits.
*/
void Converter::cmdCarCallback(const ghost::CarControl::ConstPtr& cmd_msg) {
  if((ros::Time::now() - last_pose_time_).toSec() > 0.5) 
    return;
  
  ros::Time t_now = ros::Time::now();
  
  if(first_cmd_) {
    first_cmd_ = false;
    last_cmd_time_ = t_now;
    return;
  }
  
  const double dt = (t_now - last_cmd_time_).toSec();
  
  // ax limits depend on acceleration or braking
  double new_v;
  if(cmd_msg->throttle > 0) {
    new_v = v_ + ax_max_accel_*dt*(cmd_msg->throttle - std::max(v_, 0.0)/v_max_);
  }else if(cmd_msg->throttle < 0){
    new_v = v_ + ax_max_braking_*dt*(cmd_msg->throttle - std::min(v_, 0.0)/v_max_);
  }else {
    if(v_ > 0)
      new_v = std::max(0.0, (v_ - 0.2*ax_max_braking_*dt));
    else
      new_v = std::min(0.0, (v_ + 0.2*ax_max_braking_*dt));
  }
  
  // Yaw rate comes from bicycle model
  const double new_dpsi = (new_v/l_)*tan(cmd_msg->steering*delta_max_);
    
  // Publish new twist message
  geometry_msgs::Twist twist_msg;
  twist_msg.linear.x = new_v;
  twist_msg.angular.z = new_dpsi;
  twist_pub_.publish(twist_msg);
  
  last_cmd_time_ = t_now;
}

/* main
*
* Entry point for cmd_car_to_twist
*/
int main(int argc, char** argv) {
  // Initialize
  ros::init(argc, argv, "cmd_car_to_twist");
  
  // Create the Converter object
  Converter conv;
  
  ros::spin();
}


