uint8 NONE      = 0 # Perform no amendment
uint8 SCALE     = 1 # Take current speed and multiply by this amount
uint8 OVERRIDE  = 2 # Set speed to this value

float32[] speed     # Empty if enabled == False
uint8 mode          # SCALE, RANGE, MIN, MAX, OVERRIDE
bool reverse        # if True, reverse direction around track (vehicle will drive backwards)
bool enabled