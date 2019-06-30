# Contains the state of the controls on the car and the measured motion of the wheels.

Header header

# The controls executed by the car
CarControl car_control

# Flag for if override is active (true = active)
bool override

# Encoder pulse counts for each wheel
uint32 FL_pulse_count
uint32 FR_pulse_count
uint32 RL_pulse_count
uint32 RR_pulse_count
