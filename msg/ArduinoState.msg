# Measured quantities on the Arduino
# This is intended to be published by the onbaord arduino

Header header

float32 steering       # Value of the executed steering command [-1,1]
                       # 1.0 = max left, -1.0 = max right

float32 throttle       # Value of the executed throttle command [-1,1]
                       # 1.0 = max throttle, -1.0 = max brake

uint32 FL_pulse_count  # Count of FL wheel encoder pulses
uint32 FR_pulse_count  # Count of FR wheel encoder pulses
uint32 RL_pulse_count  # Count of RL wheel encoder pulses
uint32 RR_pulse_count  # Count of RR wheel encoder pulses

bool override          # Flag for if override is active
                       # True = override active