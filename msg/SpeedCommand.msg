uint8 SCALE     = 0 # Take current speed and multiply by this amount
uint8 RANGE     = 1 # Take current speed and bound it by these values (speed[0] min, speed[1] max)
uint8 MIN       = 2 # Take current speed and ensure it is no less than this value
uint8 MAX       = 3 # Take current speed and ensure it is no more than this value
uint8 OVERRIDE  = 4 # Set speed to this value

uint8 LIN = 0 # Only consider linear velocity
uint8 ANG = 1 # Only consider angular velocity
uint8 ALL = 2 # Consider both linear and angular velocity
uint8 OFF = 3 # Disable external speed commanding

float32[] lin_speed # Empty if component == ANG or OFF
float32[] ang_speed # Empty if component == LIN or OFF
uint8 mode      # SCALE, RANGE, MIN, MAX, OVERRIDE
uint8 component # LIN, ANG, ALL, OFF