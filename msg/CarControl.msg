# Car steering and throttle commands.
# This is intended to be published by the controller, and 
# subscribed to by the arduino.

float32 steering    # Value of the steering command [-1,1]
                    # 1.0 = max left, -1.0 = max right

float32 throttle    # Value of the throttle command [-1,1]
                    # 1.0 = max throttle, -1.0 = max brake