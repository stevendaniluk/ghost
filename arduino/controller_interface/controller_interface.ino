// Test script for RC steering servoe and ESC

#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
#else
  #include <WProgram.h>
#endif

#include <ros.h>
#include <geometry_msgs/Twist.h>
#include <ghost/ServoControl.h>
#include <Servo.h> 

ros::NodeHandle  nh;

// Intantiate the servos
Servo steering;
Servo esc;

// Steering command callback
void cmdServoCallback(const ghost::ServoControl& msg) {
  steering.write(msg.steering);  // Set steering angle 
  esc.write(msg.throttle);       // Set throttle position
}

// Setup subscribers
ros::Subscriber<ghost::ServoControl> servo_sub("cmd_servo", cmdServoCallback);

void setup(){
  
  nh.initNode();
  
  nh.subscribe(servo_sub);

  while(!nh.connected()) {nh.spinOnce();}

  // Get centre servo positions from parameter server
  int steering_centre_pwm;
  int throttle_centre_pwm;
  if (!nh.getParam("steering_centre_pwm", &steering_centre_pwm)) {
    steering_centre_pwm = 127;
  }
  if (!nh.getParam("throttle_centre_pwm", &throttle_centre_pwm)) {
    throttle_centre_pwm = 0;
  }
  
  steering.attach(9);
  esc.attach(10);
  delay(10);
  steering.write(steering_centre_pwm);
  esc.write(throttle_centre_pwm);
}

void loop(){
  nh.spinOnce();

// TODO: Monitor serial connection
  
  delay(1);
}
