std_msgs/Header header                      # contains publish time
aarapsi_robot_pack/RobotDetails slam_stats  # ground truth ego,index,target,zone
aarapsi_robot_pack/RobotDetails pred_stats  # estimated ego,index,target,zone
int64 last_id                               # id of last received Label
string command_mode                         # current command mode
string safety_mode                          # current safety mode
string reject_mode                          # current reject mode
string lookahead_mode                       # current lookahead mode
int64[] zone_indices                        # list of zone indices
float64 zone_length                         # zone length [m]
int64 zone_count                            # number of zones
float64 lookahead                           # lookahead amount (qty depends on lookahead mode)