from videoStabilization import videoStabilizationReference
from testTraking import testKalmanTracking, testMedianFlowTracking
import numpy as np

# Deliver 5. Video Surveillance for Road Traffic Monitoring

# Datasets ---> highway / traffic (requires stabilization) / video2 (own dataset)
dataset_path = 'Datasets/'
sequence_path_traffic = 'Datasets/traffic/input/'

# Task 0. Compute stabilization vectors for traffic sequence
# motionVectors_traffic, point_traffic= videoStabilizationReference(sequence_path_traffic, blockSize=30, areaOfSearch=20, Backward=False, ID='traffic', point = [217,74])
# np.save('Stabilization/traffic/motionVectors_traffic', motionVectors_traffic)
# np.save('Stabilization/traffic/point_traffic', point_traffic)

# Task 1.1. Vehicle Tracker with Kalman filter
# downSampleSequence(source_path, factor_x = 0.75, factor_y= 0.75 ) # to downsample frames if necessary
testKalmanTracking(dataset_path, ID='highway', alpha=2.5, rho=0.2, holeFilling=True, areaFiltering=True, P=50, connectivity=4, Morph=True, GT_available=True, estimate_speed=False)

# Task 1.2. Vehicle Tracker with other tools
testMedianFlowTracking(dataset_path, ID='highway', alpha=2.5, rho=0.2, holeFilling=True, areaFiltering=True, P=50, connectivity=4, Morph=True, GT_available=True)

# Task 2-3.  Speed estimator + Our own study
testKalmanTracking(dataset_path, ID='highway', alpha=2.5, rho=0.2, holeFilling=True, areaFiltering=True, P=50, connectivity=4, Morph=True, GT_available=True, estimate_speed=True)







