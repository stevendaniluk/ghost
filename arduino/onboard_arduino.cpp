/* Interface for controlling the car

Monitors the cmd_car topic for commands from the controller, as well as 
monitoring inputs from the transmitter. When override is active, the 
transmitter commands are issued to the steering servo and ESC, and when
override is inactive, the controller commands are issued.

Wheel encoder signals are also monitored to count pulses.

Commands are in the range [-1,1], with 1.0 being max left, and -1.0 being max right.

Override is activated by pressing the third channel button on the remote. When active
the servos will be controled by the remote, and the cmd_car_executed topic will
contain the remote commands.

Wheel encoder signals are also monitored to count pulses.

The arduino_state topic is published, which contains the commands executed 
on the servo and esc, an override flag, and the encoder pulses.
*/

//--------------------------------------------------
// INCLUDES
//--------------------------------------------------

#if defined(ARDUINO) && ARDUINO >= 100
	#include "Arduino.h"
#else
	#include <WProgram.h>
#endif

#include <ros.h>
#include <ros/time.h>
#include <std_msgs/UInt32.h>
#include <ghost/CarControl.h>
#include <ghost/ControlState.h>
#include <Servo.h>

// Use PinChangeInt library to detect rising/falling/change on any pin
// Declare which ports will not be used to save memory
//#define NO_PORTB_PINCHANGES
#define NO_PORTC_PINCHANGES
#include "PinChangeInt.h" // https://github.com/GreyGnome/PinChangeInt

//--------------------------------------------------
// PARAMETERS
//--------------------------------------------------

// Assign pins
#define STEERING_IN_PIN 2
#define THROTTLE_IN_PIN 3
#define OVERRIDE_PIN 4
#define STEERING_OUT_PIN 5
#define THROTTLE_OUT_PIN 6
#define FL_ENCODER_PIN 9
#define FR_ENCODER_PIN 10
#define RL_ENCODER_PIN 8
#define RR_ENCODER_PIN 7

// Set PWM and servo parameters
const uint8_t steering_max = 137;           // Max steering value for arduino lirary [0,255]
const uint8_t steering_min = 63;            // Min steering value for arduino lirary [0,255]
const uint8_t steering_centre = 100;        // Centre steering value for arduino lirary [0,255]
const uint8_t throttle_max = 132;           // Max throttle value for arduino lirary [0,255]
const uint8_t throttle_min = 58;            // Min throttle value for arduino lirary [0,255]
const uint8_t throttle_centre = 87;         // Centre throttle value for arduino lirary [0,255]
const uint16_t max_pwm = 1970;              // Max detectable PWM value (from both steering servo and ESC)
const uint16_t min_pwm = 1020;              // Min detectable PWM value (from both steering servo and ESC)
const uint16_t override_pwm_thresh = 1500;  // Threshold PWM value for 3rd channel switch

// Pub/Sub settings
const int16_t pub_rate = 30;      // Rate to publish executed commands and encoder pulses
const int16_t cmd_timout = 500;   // Maximum time to wait for cmd_car message before zeroing commands

//--------------------------------------------------
// VARIABLE DECLERATION
//--------------------------------------------------

// Inbound and outbound message timing variables
const uint16_t delta_pub_millis = round(1000/pub_rate);  // Time between publishing
unsigned long prev_pub_time = 0;                         // Time of last published message
unsigned long cmd_receive_time = 0;                      // Time of last command received

// Create arduino servo objects
Servo steering;
Servo throttle;

// Output values for the servo and ESC from the controller and override
float steering_cmd;         // From the controller [-1,1]
float throttle_cmd;         // From the controller [-1,1]
float steering_override;    // From override [-1,1]
float throttle_override;    // From override [-1,1]

// Throttle and steering interrupt monitoring variables
uint16_t steering_override_pwm_local;            // Local copy of steering signal
uint16_t throttle_override_pwm_local;            // Local copy of throttle signal
volatile uint16_t steering_override_pwm_shared;  // Steering signal
volatile uint16_t throttle_override_pwm_shared;  // Throttle signal
volatile uint16_t override_pwm_shared;           // Override signal
volatile bool steering_flag_shared = false;      // When the steering signal changes
volatile bool throttle_flag_shared = false;      // When the throttle signal changes
volatile bool override_flag_shared = false;      // When the override signal changes
uint32_t steering_start;                         // Rising edge of steering servo pulse (only used in ISR)
uint32_t throttle_start;                         // Rising edge of throttle servo pulse (only used in ISR)
uint32_t override_start;                         // Rising edge of override servo pulse (only used in ISR)

// Flags for checking if override has changed
bool override_active = false;
bool prev_override_active = false;

// Determine centre throttle PWM value based on servo library values (it is not 50/50 throttle and brake)
const float throttle_frac = float(throttle_max - throttle_centre)/float(throttle_max - throttle_min);
const uint16_t throttle_centre_pwm = min_pwm + (max_pwm - min_pwm)*(1 - throttle_frac);

// Encoder variables
volatile uint32_t FL_encoder_pulses_shared = 0;
volatile uint32_t FR_encoder_pulses_shared = 0;
volatile uint32_t RL_encoder_pulses_shared = 0;
volatile uint32_t RR_encoder_pulses_shared = 0;

//--------------------------------------------------
// FUNCTION DECLARATION
//--------------------------------------------------

void cmdInCallback(const ghost::CarControl& msg);
void checkOverride();
void steeringISR();
void throttleISR();
void overrideISR();
void FLEncoderISR();
void FREncoderISR();
void RLEncoderISR();
void RREncoderISR();

//--------------------------------------------------
// SETUP
//--------------------------------------------------

ros::NodeHandle_<ArduinoHardware, 1, 2, 150, 400> nh;

// Setup subscribers
ros::Subscriber<ghost::CarControl> cmd_sub("cmd_car", cmdInCallback);

// Setup Publishers
ghost::ControlState state_msg;
ros::Publisher state_pub("arduino_state", &state_msg);

void setup(){
	nh.getHardware()->setBaud(115200);
	nh.initNode();
		
	nh.subscribe(cmd_sub);
	nh.advertise(state_pub);
	
	// Wait for connection
	while(!nh.connected()) {nh.spinOnce();}

	// Initialize variables for controls (so the first message isn't zero)
	steering_cmd = steering_centre;
	throttle_cmd = throttle_centre;
	steering_override_pwm_shared = steering_cmd;
	throttle_override_pwm_shared = throttle_cmd;
	
	// Setup the servos
	steering.attach(STEERING_OUT_PIN);
	throttle.attach(THROTTLE_OUT_PIN);
	delay(10);
	steering.write(steering_cmd);
	throttle.write(throttle_cmd);
	
	// Attach interrupt to read override and encoder signals
	PCintPort::attachInterrupt(OVERRIDE_PIN, overrideISR, CHANGE);
	PCintPort::attachInterrupt(FL_ENCODER_PIN, FLEncoderISR, CHANGE);
	PCintPort::attachInterrupt(FR_ENCODER_PIN, FREncoderISR, CHANGE);
	PCintPort::attachInterrupt(RL_ENCODER_PIN, RLEncoderISR, CHANGE);
	PCintPort::attachInterrupt(RR_ENCODER_PIN, RREncoderISR, CHANGE);
	
	delay(1000);
}// end setup

//--------------------------------------------------
// MAIN
//--------------------------------------------------

void loop() {
	// Get controller commands from callback
	nh.spinOnce();

	// Check if controller commands are being overidden
	checkOverride();
	
	// Fill the message with the controller or the override commands
	if(override_active) {    
		// Make a local copy of inputs
		noInterrupts();
		if(steering_flag_shared) {
			steering_override_pwm_local = steering_override_pwm_shared;
			steering_flag_shared = false;
		}
		if(throttle_flag_shared) {
			throttle_override_pwm_local = throttle_override_pwm_shared;
			throttle_flag_shared = false;
		}
		interrupts();
		
		// Map to range [-1,1], and make sure they are within bounds
		steering_override = -2.0*float(steering_override_pwm_local - min_pwm)/float(max_pwm - min_pwm) + 1.0;
		steering_override = min(max(steering_override,-1.0), 1.0);
		
		if(throttle_override_pwm_local >= throttle_centre_pwm){
			throttle_override = float(throttle_override_pwm_local - throttle_centre_pwm)/float(max_pwm - throttle_centre_pwm);
		}else{
			throttle_override = -float(throttle_centre_pwm - throttle_override_pwm_local)/float(throttle_centre_pwm - min_pwm);
		}
		throttle_override = min(max(throttle_override,-1.0), 1.0);

		state_msg.car_control.steering = steering_override;
		state_msg.car_control.throttle = throttle_override;
	}else {
		// Check controller timeout
		if((millis() - cmd_receive_time) > cmd_timout) {
			steering_cmd = 0.0;
			throttle_cmd = 0.0;
		}
		state_msg.car_control.steering = steering_cmd;
		state_msg.car_control.throttle = throttle_cmd;
	}

	// Write the commands to the servos (must convert to [0,255] range)
	uint16_t steering_cmd_write, throttle_cmd_write;
	if(state_msg.car_control.steering > 0.0){
		steering_cmd_write = steering_centre - state_msg.car_control.steering*(steering_centre - steering_min);
	}else{
		steering_cmd_write = steering_centre - state_msg.car_control.steering*(steering_max - steering_centre);
	}
	if(state_msg.car_control.throttle > 0.0){
		throttle_cmd_write = throttle_centre + state_msg.car_control.throttle*(throttle_max - throttle_centre);
	}else{
		throttle_cmd_write = throttle_centre + state_msg.car_control.throttle*(throttle_centre - throttle_min);
	}
	steering.write(steering_cmd_write);
	throttle.write(throttle_cmd_write);

	// Publish messages at the desired rate
	const unsigned long pub_time = millis();
	if ((pub_time - prev_pub_time) > delta_pub_millis) {
		prev_pub_time = pub_time;
				
		// Get encoder counts
		noInterrupts();
		state_msg.FL_pulse_count = FL_encoder_pulses_shared;
		state_msg.FR_pulse_count = FR_encoder_pulses_shared;
		state_msg.RL_pulse_count = RL_encoder_pulses_shared;
		state_msg.RR_pulse_count = RR_encoder_pulses_shared;
		interrupts();
						
		// Publish the messages
		state_msg.override = override_active;
		state_msg.header.stamp = nh.now();
		state_pub.publish(&state_msg);
	}
	
}// end loop

//--------------------------------------------------
// FUNCTIONS
//--------------------------------------------------

// Steering command callback for messages from the controller
void cmdInCallback(const ghost::CarControl& msg) {
	steering_cmd = msg.steering;
	throttle_cmd = msg.throttle;
	cmd_receive_time = millis();
}

// Check input servo pin for override activity
void checkOverride() {
	// Check the override value
	noInterrupts();
	if(override_flag_shared) {
		override_active = (override_pwm_shared >= override_pwm_thresh);
		override_flag_shared = false;
	}
	interrupts();
	
	// Attach/dettach the steering and throttle interrupts 
	// (don't need them running when override not in use)
	if (override_active) {
		// This is the start of an override, attach interrupts
		PCintPort::attachInterrupt(STEERING_IN_PIN, steeringISR, CHANGE); 
		PCintPort::attachInterrupt(THROTTLE_IN_PIN, throttleISR, CHANGE);
	}else if (!override_active && prev_override_active) {
		// This is the end of an override, dettachinterrupts
		PCintPort::detachInterrupt(STEERING_IN_PIN);
		PCintPort::detachInterrupt(THROTTLE_IN_PIN);
	}
	prev_override_active = override_active;
}

// Steering interrupt service routine
void steeringISR() {
	if(digitalRead(STEERING_IN_PIN) == HIGH) { 
		// It's a rising edge of the signal pulse, so record its value
		steering_start = micros();
	} else {
		// It is a falling edge, so subtract the time of the rising edge to get the pulse duration
		steering_override_pwm_shared = (uint16_t)(micros() - steering_start);
		// Set the steering flag to indicate that a new steering signal has been received
		steering_flag_shared = true;
	}
}

// Throttle interrupt service routine
void throttleISR() {
	if(digitalRead(THROTTLE_IN_PIN) == HIGH) { 
		// It's a rising edge of the signal pulse, so record its value
		throttle_start = micros();
	} else {
		// It is a falling edge, so subtract the time of the rising edge to get the pulse duration 
		throttle_override_pwm_shared = (uint16_t)(micros() - throttle_start);
		// Set the throttle flag to indicate that a new throttle signal has been received
		throttle_flag_shared = true;
	}
}

// Override interrupt service routine
void overrideISR() {
	if(digitalRead(OVERRIDE_PIN) == HIGH) { 
		// It's a rising edge of the signal pulse, so record its value
		override_start = micros();
	} else {
		// It is a falling edge, so subtract the time of the rising edge to get the pulse duration
		override_pwm_shared = (uint16_t)(micros() - override_start);
		// Set the override flag to indicate that a new override signal has been received
		override_flag_shared = true;
	}
}

// Encoder interrupt service routine
void FLEncoderISR() {
	// Simply need to count the pulses
	FL_encoder_pulses_shared++;
}

// Encoder interrupt service routine
void FREncoderISR() {
	// Simply need to count the pulses
	FR_encoder_pulses_shared++;
}

// Encoder interrupt service routine
void RLEncoderISR() {
	// Simply need to count the pulses
	RL_encoder_pulses_shared++;
}

// Encoder interrupt service routine
void RREncoderISR() {
	// Simply need to count the pulses
	RR_encoder_pulses_shared++;
}
