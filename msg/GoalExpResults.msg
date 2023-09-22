string VPR = VPR
string SVM = SVM

string mode     # VPR or SVM
int16 id
bool success
std_msgs/Header header
aarapsi_robot_pack/ExpResults localisation
aarapsi_robot_pack/xyw path_start_pos
aarapsi_robot_pack/xyw path_finish_pos
aarapsi_robot_pack/xyw robot_goal_start_pos
aarapsi_robot_pack/xyw robot_goal_finish_pos
aarapsi_robot_pack/xyw slam_goal_start_pos
aarapsi_robot_pack/xyw slam_goal_finish_pos
aarapsi_robot_pack/xyw goal_position
float64 point
float64 shoot