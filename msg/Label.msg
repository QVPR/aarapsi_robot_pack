uint8 DATA      = 0
uint8 CRUNCHER  = 1
uint8 MONITOR   = 2
uint8 FOLLOWER  = 3

sensor_msgs/CompressedImage query_image # captured image
float64[] distance_vector               # compared feature to reference set
aarapsi_robot_pack/xyw gt_ego           # from ground truth
aarapsi_robot_pack/xyw vpr_ego          # from VPR
aarapsi_robot_pack/xyw robot_ego        # from robot wheel encoders
int64 match_index                       # best vpr match
int64 truth_index                       # best groundtruth match
bool gt_class                           # whether in-tolerance of groundtruth
bool svm_class                          # whether in-tolerance of svm
float64 svm_prob                        # svm probability score
float64 svm_z                           # svm z value score
float64[] svm_factors                   # svm factors
std_msgs/Header header                  # contains last publish time
time[] stamps                           # recorded timestamp for each step
int64 id                                # unique identifier
uint8 step                              # one of DATA, CRUNCHER, MONITOR, FOLLOWER