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
    // CarControl Message
    ros::Time msg_receive_time_;
    
    // ArduinoControl Message [PWM]
    int steering_pwm_;
    int throttle_pwm_;
    
    // Car parameters
    int steering_centre_pwm_;
    int steering_max_pwm_;
    int steering_min_pwm_;
    int throttle_centre_pwm_;
    int throttle_max_pwm_;
    int throttle_min_pwm_;
    
    // Pubs/Subs
    ros::Subscriber car_cmd_sub_;
    ros::Publisher arduino_cmd_pub_;
    
    void cmdCarCallback(const ghost::CarControl& msg);
    void publishServo();

};// end Converter

void Converter::cmdCarCallback(const ghost::CarControl& msg) {
  // Record when the message was received, for timeout purposes
  msg_receive_time_ = msg.header.stamp;

  // Convert from [-1,1] to PWM values
  if(msg.steering >= 0) {
    steering_pwm_ = steering_centre_pwm_ + int(msg.steering*(steering_max_pwm_ - steering_centre_pwm_));
  }else {
    steering_pwm_ = steering_centre_pwm_ + int(msg.steering*(steering_centre_pwm_ - steering_min_pwm_));
  }
  if(msg.throttle >= 0) {
    throttle_pwm_ = throttle_centre_pwm_ + int(msg.throttle*(throttle_max_pwm_ - throttle_centre_pwm_));
  }else {
    throttle_pwm_ = throttle_centre_pwm_ + int(msg.throttle*(throttle_centre_pwm_ - throttle_min_pwm_));
  }

  // Publish commands
  publishServo();
}// end callback

void Converter::publishServo() {
  // Publish steering and throttle on cmd_arduino topic
  ghost::ArduinoControl msg;
  msg.steering = steering_pwm_;
  msg.throttle = throttle_pwm_;
  arduino_cmd_pub_.publish(msg);
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
  nh.param("steering_max_pwm", converter.steering_max_pwm_, 255);
  nh.param("steering_min_pwm", converter.steering_min_pwm_, 0);
  nh.param("throttle_centre_pwm", converter.throttle_centre_pwm_, 127);
  nh.param("throttle_max_pwm", converter.throttle_max_pwm_, 255);
  nh.param("throttle_min_pwm", converter.throttle_min_pwm_, 0);
  
  // Subscribe to cmd_car messsages with converter callback
  converter.car_cmd_sub_ = nh.subscribe("cmd_car", 1000, 
                                &Converter::cmdCarCallback, &converter);
  // Setup publisher of ArduinoControl messages
  converter.arduino_cmd_pub_ = nh.advertise<ghost::ArduinoControl>("cmd_arduino", 1000);
  
  // Set centre positions and publish
  converter.steering_pwm_ = converter.steering_centre_pwm_;
  converter.throttle_pwm_ = converter.throttle_centre_pwm_;
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
        converter.steering_pwm_ = converter.steering_centre_pwm_;
        converter.throttle_pwm_ = converter.throttle_centre_pwm_;
        converter.publishServo();
      }
    }
  }// end while
  
  return 0;
}// end main

