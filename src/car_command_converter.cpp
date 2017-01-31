/* Converts publsihed CarControl messages to ArduinoControl messages

   CarControl messages should be published by either the car controller, or
   a teleop node. Incoming CarControl messages contain the desired steering
   and throttle values, on the range [-1,1]. These are mapped to the
   appropriate PWM values, which are then published as an ArduinoControl 
   message to be processed by the arduino node.
*/

#include <ros/ros.h>
#include <ghost/CarControl.h>
#include <ghost/ArduinoControl.h>

//----------------------------------

class Converter {
  public:    
    ros::NodeHandle nh_;
    
    // Receive time for incoming messages
    ros::Time cmd_receive_time_;
    
    // Car parameters
    int steering_centre_pwm_;
    int steering_max_pwm_;
    int steering_min_pwm_;
    int throttle_centre_pwm_;
    int throttle_max_pwm_;
    int throttle_min_pwm_;
    
    // Pubs/Subs
    ros::Subscriber car_cmd_sub_;
    ros::Subscriber cmd_arduino_sub_;
    ros::Publisher cmd_arduino_pub_;
    ros::Publisher cmd_car_pub_;
    
    Converter();
    void cmdCarCallback(const ghost::CarControl& msg);
    void cmdArduinoCallback(const ghost::ArduinoControl& msg);
    void publishServo();
};// end Converter

Converter::Converter() {
  // Get relevant parameters
  nh_.param("arduino/str_centre", steering_centre_pwm_, 127);
  nh_.param("arduino/str_max", steering_max_pwm_, 255);
  nh_.param("arduino/str_min", steering_min_pwm_, 0);
  nh_.param("arduino/thr_centre", throttle_centre_pwm_, 127);
  nh_.param("arduino/thr_max", throttle_max_pwm_, 255);
  nh_.param("arduino/thr_min", throttle_min_pwm_, 0);
  
  // Subscribe to cmd_car and cmd_arduino_executed messsages
  car_cmd_sub_ = nh_.subscribe("cmd_car", 1000, &Converter::cmdCarCallback, this);
  cmd_arduino_sub_ = nh_.subscribe("cmd_arduino_executed", 1000, &Converter::cmdArduinoCallback, this);
  
  // Setup publishers of ArduinoControl and CarControl messages
  cmd_arduino_pub_ = nh_.advertise<ghost::ArduinoControl>("cmd_arduino", 1000);
  cmd_car_pub_ = nh_.advertise<ghost::CarControl>("cmd_car_executed", 1000);
  
  // Set centre positions and publish
  ghost::ArduinoControl msg;
  msg.steering = steering_centre_pwm_;
  msg.throttle = throttle_centre_pwm_;
  cmd_arduino_pub_.publish(msg);
  
}// end constructor

void Converter::cmdCarCallback(const ghost::CarControl& msg_in) {
  // Record when the message was received, for timeout purposes
  cmd_receive_time_ = msg_in.header.stamp;

  // Convert from [-1,1] to PWM values
  ghost::ArduinoControl msg_out;
  if(msg_in.steering >= 0) {
    msg_out.steering = steering_centre_pwm_ + int(msg_in.steering*(steering_max_pwm_ - steering_centre_pwm_));
  }else {
    msg_out.steering = steering_centre_pwm_ + int(msg_in.steering*(steering_centre_pwm_ - steering_min_pwm_));
  }
  if(msg_in.throttle >= 0) {
    msg_out.throttle = throttle_centre_pwm_ + int(msg_in.throttle*(throttle_max_pwm_ - throttle_centre_pwm_));
  }else {
    msg_out.throttle = throttle_centre_pwm_ + int(msg_in.throttle*(throttle_centre_pwm_ - throttle_min_pwm_));
  }
  
  // Publish commands
  cmd_arduino_pub_.publish(msg_out);
}// end callback

void Converter::cmdArduinoCallback(const ghost::ArduinoControl& msg_in) {
  // Convert from PWM to [-1,1] values
  ghost::CarControl msg_out;
  if(msg_in.steering >= steering_centre_pwm_) {
    msg_out.steering = float(msg_in.steering - steering_centre_pwm_)/(steering_max_pwm_ - steering_centre_pwm_);
  }else {
    msg_out.steering = float(msg_in.steering - steering_centre_pwm_)/(steering_centre_pwm_ - steering_min_pwm_);
  }
  if(msg_in.throttle >= throttle_centre_pwm_) {
    msg_out.throttle = float(msg_in.throttle - throttle_centre_pwm_)/(throttle_max_pwm_ - throttle_centre_pwm_);
  }else {
    msg_out.throttle = float(msg_in.throttle - throttle_centre_pwm_)/(throttle_centre_pwm_ - throttle_min_pwm_);
  }
  msg_out.header.stamp = ros::Time::now();
  
  // Publish executed commands
  cmd_car_pub_.publish(msg_out);
}// end callback

int main(int argc, char **argv) {
  // Initialize
  ros::init(argc, argv, "car_command_converter");
    
  // Create the converter
  Converter converter;
  
  ros::Time loop_timer = ros::Time::now();
  
  // Spin (listen to commands, convert, then publish)
  while (ros::ok()) {
    ros::spinOnce();
    
    // Default to centred commands when messages are not received in time
    if (ros::Time::now() > loop_timer + ros::Duration(0.2)) {
      loop_timer = ros::Time::now();
      if (converter.cmd_receive_time_ < (ros::Time::now() - ros::Duration(1.0))) {
        ghost::ArduinoControl msg;
        msg.steering = converter.steering_centre_pwm_;
        msg.throttle = converter.throttle_centre_pwm_;
        converter.cmd_arduino_pub_.publish(msg);
      }
    }
  }// end while
  
  return 0;
}// end main

