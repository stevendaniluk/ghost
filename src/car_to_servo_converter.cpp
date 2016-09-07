/* Converts publsihed CarControl messages to ServoControl messages

   CarControl messages should be published by either the car controller, or
   a teleop node. The desired velocity and steering angle are converted to 
   the appropriate PWM values, which are then published as a ServoControl 
   message to be processed by the arduino node.
*/

#include <ros/ros.h>
#include <ghost/CarControl.h>
#include <ghost/ServoControl.h>

//----------------------------------

class Converter {
  public:
    // CarControl Message
    float steering_angle_;
    float steering_rate_;
    float velocity_;
    float acceleration_;
    ros::Time msg_receive_time_;
    
    // ServoControl Message [PWM]
    int steering_;
    int throttle_;
    
    // Car parameters
    int steering_centre_pwm_;
    int throttle_centre_pwm_;
    int throttle_max_pwm_;
    int pwm_per_degree_;
    double vel_max_;
    
    // Pubs/Subs
    ros::Subscriber car_cmd_sub_;
    ros::Publisher servo_cmd_pub_;
    
    void cmdCarCallback(const ghost::CarControl& msg);
    void convertToServo();
    void publishServo();

};// end Converter

void Converter::cmdCarCallback(const ghost::CarControl& msg) {
  steering_angle_ = msg.steering_angle;
  steering_rate_ = msg.steering_rate;
  velocity_ = msg.velocity;
  acceleration_ = msg.acceleration;
  msg_receive_time_ = msg.header.stamp;
  
  convertToServo();
  publishServo();
}// end callback

void Converter::convertToServo() {
  // Convert from angles to PWM
  steering_ = steering_centre_pwm_ + steering_angle_*pwm_per_degree_;
  if (acceleration_ < 0) {
    // Braking
    throttle_ =  (1 + acceleration_)*throttle_centre_pwm_;
  } else {
    // Positive throttle command
    throttle_ = throttle_centre_pwm_ + 
              (velocity_/vel_max_)*(throttle_max_pwm_ - throttle_centre_pwm_);
  }
  
}// end convertToServo

void Converter::publishServo() {
  // Publish steering and throttle on cmd_servo topic
  ghost::ServoControl msg;
  msg.steering = steering_;
  msg.throttle = throttle_;
  servo_cmd_pub_.publish(msg);
}// end convertToServo


int main(int argc, char **argv) {
  // Initialize
  ros::init(argc, argv, "car_to_servo_converter");
  
  // Make our node handle
  ros::NodeHandle nh;
  
  // Create the converter
  Converter converter;
  
  // Get relevant parameters
  nh.param("steering_centre_pwm", converter.steering_centre_pwm_, 127);
  nh.param("throttle_centre_pwm", converter.throttle_centre_pwm_, 127);
  nh.param("throttle_max_pwm", converter.throttle_max_pwm_, 255);
  nh.param("pwm_per_degree", converter.pwm_per_degree_, 1);
  nh.param("vel_max", converter.vel_max_, 10.0);
  
  // Subscribe to cmd_car messsages with converter callback
  converter.car_cmd_sub_ = nh.subscribe("cmd_car", 1000, 
                                &Converter::cmdCarCallback, &converter);
  // Setup publisher of cmd_servo messages
  converter.servo_cmd_pub_ = nh.advertise<ghost::ServoControl>("cmd_servo", 1000);
  
  // Set centre positions and publish
  converter.steering_ = converter.steering_centre_pwm_;
  converter.throttle_ = converter.throttle_centre_pwm_;
  converter.msg_receive_time_ = ros::Time::now();
  converter.publishServo();
  
  ros::Time loop_timer = ros::Time::now();
  
  // Spin (listen to commands, convert, then publish)
  while (ros::ok()) {
    ros::spinOnce();
    
    // Default to centred commands when messages are not received in time
    if (ros::Time::now() > loop_timer + ros::Duration(0.2)) {
      loop_timer = ros::Time::now();
      if (converter.msg_receive_time_ < (ros::Time::now() - ros::Duration(1.0))) {
        converter.steering_ = converter.steering_centre_pwm_;
        converter.throttle_ = converter.throttle_centre_pwm_;
        converter.publishServo();
      }
    }
  }// end while
  
  return 0;
}// end main

