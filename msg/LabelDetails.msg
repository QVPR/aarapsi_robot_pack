# note: this structure assumes the publisher/s and subscriber/s have pre-existing knowledge
# about what reference data to use and have a copy of the reference data they can access already

geometry_msgs/Point odom                # x,y,z queryImage was recorded at (check if groundtruth enabled against trueId)
float64[] dvc                           # distance vector; length matches the length of reference data
int64 matchId                           # Index of VPR match
int64 trueId                            # Index ground truth thinks is best (-1 if no groundtruth)
int8 state                              # int matching 0: unknown, 1: bad, 2: good
bool compressed                         # Bool whether or not this msg struct belongs to a compressed query image
string matchPath                        # Full file path to matched reference image

## Note: Indices are with respect to the reference data where the first item is index 0.