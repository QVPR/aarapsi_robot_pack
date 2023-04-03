# note: this structure assumes the publisher/s and subscriber/s have pre-existing knowledge
# about what reference data to use and have a copy of the reference data they can access already

sensor_msgs/Image queryImage	        # Query image
aarapsi_robot_pack/LabelDetails data    # Label details (indices, dvc, ...)
std_msgs/Header header		            # Header
float64 mState			                # Continuous monitor state estimate 
float64 prob			                # Monitor probability estimate
bool mStateBin			                # Binary monitor state estimate
float64[] factors		                # two-element array of factors 1 and 2, in order.
