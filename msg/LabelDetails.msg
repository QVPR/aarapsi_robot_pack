# note: this structure assumes the publisher/s and subscriber/s have pre-existing knowledge
# about what reference data to use and have a copy of the reference data they can access already

float64[] dvc                           # distance vector; length matches the length of reference data
aarapsi_robot_pack/xyw gt_ego           # x,y,yaw queryImage was recorded at (check if groundtruth enabled against trueId)
aarapsi_robot_pack/xyw vpr_ego          # x,y,yaw as generated from queryImage
int64 matchId                           # Index of VPR match
int64 trueId                            # Index ground truth thinks is best (-1 if no groundtruth)
int8 gt_state                           # int matching -1: unknown, 0: bad, 1: good
float64 gt_error			# float value for the error as per the tolerancing scheme
## Note: Indices are with respect to the reference data where the first item is index 0.
