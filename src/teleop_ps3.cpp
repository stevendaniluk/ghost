/* Reads in PS3 controller commands, and publishes car control commands

   Reads in steering angle from the left joystick, throttle from the right
   trigger, and brake from the left trigger, then publishes it as
   a CarControl message. When throttle, brake, and steering angle are all zero,
   nothing is published.
*/

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Joy.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <ghost/CarControl.h>

//------------------------------------

class Teleop {
  public:
    Teleop();

  private:
    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    void publish();
    
    ros::NodeHandle nh_;

    ros::Publisher car_control_pub_;
    ros::Subscriber joy_sub_;
    
    // Message to be published
    ghost::CarControl msg_;
    bool publish_once_;
    
    // Axis indices for PS3 controller
    int gas_trigger_;
    int brake_trigger_;
    int steering_axis_;
    
    // For controllin the publish rate of outgoing messages
    boost::mutex publish_mutex_;
    ros::Timer timer_;
};

// Constructor
Teleop::Teleop() {

  publish_once_ = true;
  
  // Indices for a PS3 controller
  gas_trigger_ = 13;
  brake_trigger_ = 12;
  steering_axis_ = 0;
    
  // Setup pubs and subs
  joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &Teleop::joyCallback, this);
  car_control_pub_ = nh_.advertise<ghost::CarControl>("cmd_car", 1, true);
  
  // For controlling the rate of published messages
  timer_ = nh_.createTimer(ros::Duration(0.1), boost::bind(&Teleop::publish, this));
}

// Read in the joystick data
void Teleop::joyCallback(const sensor_msgs::Joy::ConstPtr& joy) {  
  msg_.steering = joy->axes[steering_axis_]; 
  msg_.throttle = -joy->axes[gas_trigger_] - joy->axes[brake_trigger_];
}

// Publish the message at a fixed rate (but don't keep publishing zero messages)
void Teleop::publish() {
  boost::mutex::scoped_lock lock(publish_mutex_);
  
  if (msg_.steering == 0 && msg_.throttle == 0) {
    // Publish once zero message, then stop
    if (publish_once_) {
      car_control_pub_.publish(msg_);
      publish_once_ = false;
    }
  } else {
    // There are controls to publish
    car_control_pub_.publish(msg_);
    publish_once_ = true;
    ROS_DEBUG("Steering=%.2f, Throttle=%.2f", msg_.steering, msg_.throttle);
  }
}

//------------------------------------

int main(int argc, char** argv) {
  ros::init(argc, argv, "teleop_ps3");
  Teleop teleop;
  
  puts(" ");
  puts("---------------------------");
  puts("Reading from PS3 controller");
  puts(" ");
  puts("Left Trigger = Brake");
  puts("Right Trigger = Throttle");
  puts("Left Joystick = Steering");

  ros::spin();
}
