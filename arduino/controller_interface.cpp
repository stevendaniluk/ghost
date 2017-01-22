/* Interface for controlling the car servos

Monitors the cmd_arduino topic for commands from the controller, as well as 
monitoring inputs from the remote to handle override commands. The 
cmd_arduino_executed topic is published, which contains the commands executed 
on the servos.

Override is activated by pressing the third channel button on the remote. When active
the servos will be controled by the remote, and the cmd_arduino_executed topic will
contain the remote commands with the override flag set true.

Motor RPM is measured and has a moving average applied to it. The RPm is included
in the cmd_arduino_executed topic.

Notes about override servo signals:
  -Range of signals is [1050, 2000]
  -Servo signals get mapped between the min and max values loaded from the param server
  -Channel three switches between 1050 (not active) and 2000 (active)

*/
//--------------------------------------------------

#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
#else
  #include <WProgram.h>
#endif

#include <ros.h>
#include <std_msgs/Float32.h>
#include <ghost/ArduinoControl.h>
#include <Servo.h> 

// Use PinChangeInt library to detect rising/falling/change on any pin
// Declare which ports will not be used to save memory
#define NO_PORTB_PINCHANGES
#define NO_PORTC_PINCHANGES
#include "PinChangeInt.h" // https://github.com/GreyGnome/PinChangeInt

// Assign servo pins
#define STEERING_IN_PIN 2
#define THROTTLE_IN_PIN 3
#define OVERRIDE_PIN 4
#define RPM_PIN 5
#define STEERING_OUT_PIN 9
#define THROTTLE_OUT_PIN 10

// Variables for interrupts monitoring
volatile uint16_t override_in_shared;     // Override signal
uint32_t steering_start;                  // Rising edge of steering servo pulse
uint32_t throttle_start;                  // Rising edge of throttle servo pulse
uint32_t override_start;                  // Rising edge of override servo pulse
bool override_flag = false;               // When the override signal changes

// Flags for checking if override has changed
bool override_active = false;
bool prev_override_active = false;

// RPM monitoring variables
volatile uint16_t rpm_pulses_shared = 0;  // Counter for pulses between average updates
const uint8_t rpm_avg_n = 10;             // Number of points to use in moving average
float rpm_readings[rpm_avg_n] = {0};      // Array or rpm readings
uint8_t rpm_index = 0;                    // Index of the current rpm reading

// Varaibles for publishing at the desired rate
const uint8_t pub_freq = 50;
const uint16_t delta_pub_millis = uint16_t round(1000/pub_freq);
unsigned long prev_pub_time = 0;

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
volatile uint16_t steering_override;
volatile uint16_t throttle_override;

//--------------------------------------------------
void cmdInCallback(const ghost::ArduinoControl& msg);
void checkOverride();
void getSteering();
void getThrottle();
void getOverride();
void getRPM();
//--------------------------------------------------

ros::NodeHandle nh;

// Setup subscribers
ros::Subscriber<ghost::ArduinoControl> cmd_in_sub("cmd_arduino", cmdInCallback);

// Setup Publishers
ghost::ArduinoControl cmd_out_msg;
ros::Publisher cmd_out_pub("cmd_arduino_executed", &cmd_out_msg);
std_msgs::Float32 rpm_msg;
ros::Publisher rpm_pub("/motor_rpm", &rpm_msg);

void setup(){
  nh.getHardware()->setBaud(115200);
  nh.initNode();
    
  nh.subscribe(cmd_in_sub);
  nh.advertise(cmd_out_pub);
  nh.advertise(rpm_pub);
  
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
  
  // Attach interrupt to read override and rpm signals
  PCintPort::attachInterrupt(OVERRIDE_PIN, getOverride, CHANGE);
  PCintPort::attachInterrupt(RPM_PIN, getRPM, CHANGE);

}// end setup

//--------------------------------------------------

void loop() {
  
  // Get controller commands from callback
  nh.spinOnce();

  // Check if controller commands are being overidden
  checkOverride();

  // Fill the message with the controller or the override commands
  if(override_active) {
    // Use the override commands
    cmd_out_msg.steering = steering_override;
    cmd_out_msg.throttle = throttle_override;
    cmd_out_msg.override = true;    
  }else {
    // Use the controller commands
    cmd_out_msg.steering = steering_ctrl;
    cmd_out_msg.throttle = throttle_ctrl;
    cmd_out_msg.override = false;
  }
  
  // Write the commands to the servos
  steering.write(cmd_out_msg.steering); 
  throttle.write(cmd_out_msg.throttle);

  // Publish messages at the desired rate
  unsigned long pub_time = millis();
  if ((pub_time - prev_pub_time) > delta_pub_millis) {
    prev_pub_time = pub_time;

    // Determine motor RPM from the pulses since the last message
    // and add it to the array of readings
    // (must divide by two, since each interrupt is 1/2 of a rotation)
    unsigned long rpm_pulses = rpm_pulses_shared;
    rpm_pulses_shared = 0;
    rpm_readings[rpm_index] = (0.5f*float(rpm_pulses)*60.0f)/(float(pub_freq)/1000.0f);

    // Compute the RPM moving average
    uint8_t oldest_index = (rpm_index + 1)%rpm_avg_n;
    float new_reading = rpm_readings[rpm_index];
    float oldest_reading = rpm_readings[oldest_index];
    rpm_msg.data = rpm_msg.data + new_reading/rpm_avg_n - oldest_reading/rpm_avg_n;
    
    // Update the readings index
    rpm_index++;
    if(rpm_index == rpm_avg_n)
      rpm_index = 0;
    
    // Publish the messages
    cmd_out_pub.publish(&cmd_out_msg);
    rpm_pub.publish(&rpm_msg);
  }
  
}// end main

//--------------------------------------------------

// Steering command callback for messages from the controller
void cmdInCallback(const ghost::ArduinoControl& msg) {
  steering_ctrl = msg.steering;
  throttle_ctrl = msg.throttle;
}

// Check input servo pin for override activity
void checkOverride() {
  // Check the override value
  if(override_flag) {
    if(override_in_shared < 1500) {
      override_active = false;
    }else {
      override_active = true;
    }
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
}// end checkOverride

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

// RPM interrupt service routine
void getRPM() {
  // Simply need to count the pulses
  // Signal changes from high to low every 1/2 turn
  rpm_pulses_shared++;
}
