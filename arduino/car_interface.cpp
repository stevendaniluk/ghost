/* Interface for controlling the car

Monitors incoming controls from the transmitter as well as the incoming topic, and switches the
active control input based on the transmitter's 3rd channel input. Encoder pules for each wheel are
also monitored.

Override is activated by pressing the third channel button on the remote. When override is active
the transmitter commands are issued to the steering servo and ESC, and when override is inactive,
the controller commands (from the incoming topic) are issued.
*/

#include <Servo.h>
#include <ghost/CarControl.h>
#include <ghost/ControlState.h>
#include <ros.h>
#include <ros/time.h>
#include <std_msgs/UInt32.h>
#include "Arduino.h"
#include "interrupt_pwm_signal.h"
#include "signal_map.h"
#include "simple_moving_average.h"

// Use PinChangeInt library to detect rising/falling/change on any pin
// Declare which ports will not be used to save memory
#define NO_PORTC_PINCHANGES
#include "PinChangeInt.h"  // https://github.com/GreyGnome/PinChangeInt

//--------------------------------------------------
// PARAMETERS AND VARIABLE DECLERATION
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

// Create value maps for PWM inputs and servo outputs, these dictate the min, center, and max
// positions for the servo positions and incoming PWM signals Argument order is: min, center, max.
const SignalMap steering_servo_map(63, 100, 137);
const SignalMap throttle_servo_map(58, 87, 132);
const SignalMap steering_pwm_map(1020, 1485, 1965);
const SignalMap throttle_pwm_map(1020, 1392, 1965);

// Threshold PWM value for 3rd channel switch to trigger override mode. On powerup the signal will
// be ~1000.
const uint16_t override_pwm_thresh = 1500;

// Outgoing state message timing
const int16_t pub_rate = 30;
const uint16_t delta_pub_millis = round(1000 / pub_rate);
unsigned long prev_pub_time = 0;

// Incoming commands
ghost::CarControl prev_cmd;
unsigned long cmd_receive_time = 0;
const int16_t cmd_timout = 500;

// Flags for tracking the override state
bool override_active = false;
bool prev_override_active = false;

// Helpers to compute a moving average of the incoming controls from the transmitter, since the
// signals can be noisy
SimpleMovingAverage<uint16_t, 5> steering_sma(steering_pwm_map.center);
SimpleMovingAverage<uint16_t, 5> throttle_sma(throttle_pwm_map.center);

// Arduino servo objects for controlling the steering servo and ESC
Servo steering;
Servo throttle;

// Helpers for interrupt signals from the transmitter
InterruptPWMSignal steering_input(steering_pwm_map.center);
InterruptPWMSignal throttle_input(throttle_pwm_map.center);
InterruptPWMSignal override_input(0);

// Pulse counts from the encoders
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

void setup() {
    nh.getHardware()->setBaud(115200);
    nh.initNode();

    nh.subscribe(cmd_sub);
    nh.advertise(state_pub);

    // Wait for connection
    while (!nh.connected()) {
        nh.spinOnce();
    }

    // Setup the servos
    steering.attach(STEERING_OUT_PIN);
    throttle.attach(THROTTLE_OUT_PIN);
    delay(10);
    steering.write(steering_servo_map.center);
    throttle.write(throttle_servo_map.center);

    // Attach interrupt to read override and encoder signals
    PCintPort::attachInterrupt(OVERRIDE_PIN, overrideISR, CHANGE);
    PCintPort::attachInterrupt(FL_ENCODER_PIN, FLEncoderISR, CHANGE);
    PCintPort::attachInterrupt(FR_ENCODER_PIN, FREncoderISR, CHANGE);
    PCintPort::attachInterrupt(RL_ENCODER_PIN, RLEncoderISR, CHANGE);
    PCintPort::attachInterrupt(RR_ENCODER_PIN, RREncoderISR, CHANGE);

    nh.loginfo("Completed car interface setup!");

    delay(1000);
}  // end setup

//--------------------------------------------------
// MAIN
//--------------------------------------------------

void loop() {
    // Get controller commands from callback
    nh.spinOnce();

    // Check if controller commands are being overidden
    checkOverride();

    // Fill the message with the controller or the override commands
    ghost::CarControl cmd_execute;
    if (override_active) {
        // Disable the interrupts to record the latest steering and throttle signals
        noInterrupts();
        steering_input.processNewSignals();
        throttle_input.processNewSignals();
        interrupts();

        // Apply the moving average to smooth the signals
        steering_sma.addDataPoint(steering_input.pwm_local);
        throttle_sma.addDataPoint(throttle_input.pwm_local);

        // Map the pwm inputs to the [-1, 1] range for controls
        cmd_execute.steering = -steering_pwm_map.mapToUnitOutput(steering_sma.getAverage());
        cmd_execute.throttle = throttle_pwm_map.mapToUnitOutput(throttle_sma.getAverage());
    } else {
        // Check controller timeout
        if ((millis() - cmd_receive_time) > cmd_timout) {
            prev_cmd.steering = 0.0;
            prev_cmd.throttle = 0.0;
        }
        cmd_execute = prev_cmd;
    }

    // Write the commands to the servos (must convert to [0,255] range)
    const uint16_t steering_cmd_write = steering_servo_map.mapFromUnitInput(-cmd_execute.steering);
    const uint16_t throttle_cmd_write = throttle_servo_map.mapFromUnitInput(cmd_execute.throttle);

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
        state_msg.car_control = cmd_execute;
        state_msg.override = override_active;
        state_msg.header.stamp = nh.now();
        state_pub.publish(&state_msg);
    }

}  // end loop

//--------------------------------------------------
// FUNCTIONS
//--------------------------------------------------

// Callback for command messages from the controller
void cmdInCallback(const ghost::CarControl& msg) {
    prev_cmd = msg;
    cmd_receive_time = millis();
}

// Check input servo pin for override activity
void checkOverride() {
    // Disable the interrupts to record the latest override signal
    override_input.processNewSignalsNoInterrupts();

    override_active = (override_input.pwm_local <= override_pwm_thresh);

    // Attach/dettach the steering and throttle interrupts when needed
    if (override_active && !prev_override_active) {
        // This is the start of an override, attach interrupts
        PCintPort::attachInterrupt(STEERING_IN_PIN, steeringISR, CHANGE);
        PCintPort::attachInterrupt(THROTTLE_IN_PIN, throttleISR, CHANGE);
    } else if (!override_active && prev_override_active) {
        // This is the end of an override, dettachinterrupts
        PCintPort::detachInterrupt(STEERING_IN_PIN);
        PCintPort::detachInterrupt(THROTTLE_IN_PIN);
    }
    prev_override_active = override_active;
}

// Steering interrupt service routine
void steeringISR() {
    if (digitalRead(STEERING_IN_PIN) == HIGH) {
        steering_input.markSignalStart();
    } else {
        steering_input.markSignalEnd();
    }
}

// Throttle interrupt service routine
void throttleISR() {
    if (digitalRead(THROTTLE_IN_PIN) == HIGH) {
        throttle_input.markSignalStart();
    } else {
        throttle_input.markSignalEnd();
    }
}

// Override interrupt service routine
void overrideISR() {
    if (digitalRead(OVERRIDE_PIN) == HIGH) {
        override_input.markSignalStart();
    } else {
        override_input.markSignalEnd();
    }
}

// FL wheel encoder interrupt to increment pulses
void FLEncoderISR() {
    FL_encoder_pulses_shared++;
}

// FR wheel encoder interrupt to increment pulses
void FREncoderISR() {
    FR_encoder_pulses_shared++;
}

// RL wheel encoder interrupt to increment pulses
void RLEncoderISR() {
    RL_encoder_pulses_shared++;
}

// RR wheel encoder interrupt to increment pulses
void RREncoderISR() {
    RR_encoder_pulses_shared++;
}
