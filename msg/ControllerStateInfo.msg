# Generic message structure for testing purposes

std_msgs/Header header

uint8[] query_image 				# at top bc this is big and floods screen
aarapsi_robot_pack/LabelDetails label_details # at top bc this.dvc is big and floods screen

aarapsi_robot_pack/SafetyStates safety_states

float64 current_yaw
float64 target_yaw
float64 true_yaw
float64 delta_yaw

float64 lookahead
string lookahead_mode

# extracted from MonitorDetails
float64 mState
float64 prob
bool mStateBin
float64[] factors

string vpr_topic
string jackal_topic
string groundtruth_topic


