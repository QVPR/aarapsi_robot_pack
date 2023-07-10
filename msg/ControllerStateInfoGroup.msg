# extracted from LabelDetails:
aarapsi_robot_pack/xyw gt_ego           # x,y,yaw queryImage was recorded at (check if groundtruth enabled against trueId)
aarapsi_robot_pack/xyw vpr_ego          # x,y,yaw as generated from queryImage
int64 matchId                           # Index of VPR match
int64 trueId                            # Index ground truth thinks is best (-1 if no groundtruth)
int8 gt_state                           # int matching -1: unknown, 0: bad, 1: good
float64 gt_error			# float value for error in groundtruth scheme

string safety_mode
string command_mode

float64 current_yaw
float64 target_yaw
float64 true_yaw
float64 delta_yaw

float64 lookahead
string lookahead_mode

# extracted from MonitorDetails:
float64 mState
float64 prob
bool mStateBin
float64[] factors
