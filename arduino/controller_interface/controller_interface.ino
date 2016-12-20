/* Interface for controlling the car servos

Monitors the cmd_servo topic for commands from the controller, as well as 
monitoring inputs from the remote to handle override commands. The cmd_servo_executed
topic is published, which contains the commands executed on the servos.

Override is activated by pressing the third channel button on the remote. When active
the servos will be controled by the remote, and the cmd_servo_executed topic will contain
the remote commands with the override flag set true.

Notes about override servo signals:
  -Range of signals is [1050, 2000]
  -Servo signals get mapped between the min and max values loaded from the param server
  -Channel three switches between 1050 (no active) and 2000 (active)

*/
//--------------------------------------------------

#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
#else
  #include <WProgram.h>
#endif

#include <ros.h>
#include <ros/time.h>
#include <std_msgs/UInt16.h>
#include <ghost/ServoControl.h>
#include <Servo.h> 
#include "PinChangeInt.h" // https://github.com/GreyGnome/PinChangeInt

// Assign servo pins
#define STEERING_IN_PIN 2
#define THROTTLE_IN_PIN 3
#define OVERRIDE_PIN 4
#define STEERING_OUT_PIN 9
#define THROTTLE_OUT_PIN 10

// Variables for interrupts monitoring
volatile uint16_t override_in_shared;     // Override signal
uint32_t steering_start;                  // Rising edge of steering servo pulse
uint32_t throttle_start;                       // Rising edge of throttle servo pulse
uint32_t override_start;                  // Rising edge of override servo pulse
bool override_flag = false;               // When the override signal changes

bool override_active = false;
bool prev_override_active = false;

// Servo parameters
Servo steering;
Servo throttle;
int16_t steering_max_pwm;
int16_t steering_min_pwm;
int16_t throttle_max_pwm;
int16_t throttle_min_pwm;

// Output values for the servos from the controller and override
uint16_t steering_ctrl;
uint16_t throttle_ctrl;
uint16_t steering_override;
uint16_t throttle_override;

//--------------------------------------------------

// Steering command callback for messages from the controller
void cmdServoCallback(const ghost::ServoControl& msg) {
  steering_ctrl = msg.steering;
  throttle_ctrl = msg.throttle;
}

// Check input servo pin for override activity
void checkOverride() {
  // Check the value
  if(override_flag) {
    noInterrupts();
    if(override_in_shared < 1500) {
      override_active = false;
    }else {
      override_active = true;
    }
    interrupts();
  }
  
  // Attach/dettach the steering and throttle interrupts 
  // (don't need them running when override not in use)
  if (override_active) {
    // This is the start of an override, attach interrupts
    PCintPort::attachInterrupt(STEERING_IN_PIN, getSteering, CHANGE); 
    PCintPort::attachInterrupt(THROTTLE_IN_PIN, getThrottle, CHANGE);
  }else if (!override_active && prev_override_active) {
    // This is the end of an override, dettachinterrupts
    PCintPort::detachInterrupt(STEERING_IN_PIN);
    PCintPort::detachInterrupt(THROTTLE_IN_PIN);
  }
  prev_override_active = override_active;
}

//--------------------------------------------------

ros::NodeHandle  nh;

// Setup subscribers
ros::Subscriber<ghost::ServoControl> servo_cmd_sub("cmd_servo", cmdServoCallback);

// Setup Publishers
ghost::ServoControl cmd_servo_executed;
ros::Publisher cmd_servo_executed_pub("cmd_servo_executed", &cmd_servo_executed);

void setup(){
  nh.initNode();
  
  nh.subscribe(servo_cmd_sub);
  nh.advertise(cmd_servo_executed_pub);
  
  // Wait for connection
  while(!nh.connected()) {nh.spinOnce();}

  // Get servo positions from parameter server
  int16_t steering_centre_pwm;
  int16_t throttle_centre_pwm;

  if (!nh.getParam("steering_centre_pwm", &steering_centre_pwm)) {
    steering_centre_pwm = 100;
  }
  if (!nh.getParam("steering_max_pwm", &steering_max_pwm)) {
    steering_max_pwm = 140;
  }
  if (!nh.getParam("steering_min_pwm", &steering_min_pwm)) {
    steering_min_pwm = 60;
  }
  if (!nh.getParam("throttle_centre_pwm", &throttle_centre_pwm)) {
    throttle_centre_pwm = 87;
  }
  if (!nh.getParam("throttle_max_pwm", &throttle_max_pwm)) {
    throttle_max_pwm = 135;
  }
  if (!nh.getParam("throttle_min_pwm", &throttle_min_pwm)) {
    throttle_min_pwm = 53;
  }

  // Initialize variables for controls (so the first message isn't zero)
  steering_ctrl = steering_centre_pwm;
  throttle_ctrl = throttle_centre_pwm;
  steering_override = steering_centre_pwm;
  throttle_override = throttle_centre_pwm;
  
  // Setup the servos
  steering.attach(STEERING_OUT_PIN);
  throttle.attach(THROTTLE_OUT_PIN);
  delay(10);
  steering.write(steering_centre_pwm);
  throttle.write(throttle_centre_pwm);
  
  // Attach interrupt to read override signal
  PCintPort::attachInterrupt(OVERRIDE_PIN, getOverride, CHANGE);
}

//--------------------------------------------------

void loop() {
  // Get controller commands from callback
  nh.spinOnce();
  
  // Check if controller commands are being overidden
  checkOverride();

  // Fill the message with the controller or the override
  if(override_active) {
    // Use the override commands
    cmd_servo_executed.steering = steering_override;
    cmd_servo_executed.throttle = throttle_override;
    cmd_servo_executed.override = true;    
  }else {
    // Use the controller commands
    cmd_servo_executed.steering = steering_ctrl;
    cmd_servo_executed.throttle = throttle_ctrl;
    cmd_servo_executed.override = false;
  }
  cmd_servo_executed.header.stamp = nh.now();
  cmd_servo_executed_pub.publish(&cmd_servo_executed);
  
  // Write the commands to the servos
  steering.write(cmd_servo_executed.steering); 
  throttle.write(cmd_servo_executed.throttle);
  
  delay(50);
}

//--------------------------------------------------

// Steering interrupt service routine
void getSteering() {
  if(digitalRead(STEERING_IN_PIN) == HIGH) { 
    // It's a rising edge of the signal pulse, so record its value
    steering_start = micros();
  } else {
    // It is a falling edge, so subtract the time of the rising edge to get the pulse duration
    steering_override = (uint16_t)(micros() - steering_start);
    // Map to proper range, and make sure it is within bounds
    steering_override = map(steering_override, 1050, 2000, steering_min_pwm, steering_max_pwm);
    steering_override = constrain(steering_override, steering_min_pwm, steering_max_pwm);
  }
}

// Throttle interrupt service routine
void getThrottle() {
  if(digitalRead(THROTTLE_IN_PIN) == HIGH) { 
    // It's a rising edge of the signal pulse, so record its value
    throttle_start = micros();
  } else {
    // It is a falling edge, so subtract the time of the rising edge to get the pulse duration 
    throttle_override = (uint16_t)(micros() - throttle_start);
    // Map to proper range, and make sure it is within bounds
    throttle_override = map(throttle_override, 1050, 2000, throttle_min_pwm, throttle_max_pwm);
    throttle_override = constrain(throttle_override, throttle_min_pwm, throttle_max_pwm);
  }
}

// Override interrupt service routine
void getOverride() {
  if(digitalRead(OVERRIDE_PIN) == HIGH) { 
    // It's a rising edge of the signal pulse, so record its value
    override_start = micros();
    override_flag = false;
  } else {
    // It is a falling edge, so subtract the time of the rising edge to get the pulse duration
    override_in_shared = (uint16_t)(micros() - override_start);
    // Set the override flag to indicate that a new override signal has been received
    override_flag = true;
  }
}
