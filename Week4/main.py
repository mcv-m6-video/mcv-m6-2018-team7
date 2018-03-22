from opticalFlowEstimation import estimateOpticalFlow, optimalBlockSizeArea, estimateOpticalFlowFarneback, estimateOpticalFlowTVL1
from opticalFlowEvaluation import evaluateOpticalFlow
from readInputWriteOutput import plotMSENPEPNCurves, plotOpticalFlowHSV, convertVideotoFrames, precisionRecallCurveDataset
from videoStabilization import videoStabilizationReference, videoStabilizationMovement, videoStabilizationPairs, HomographyTransformStab
from backgroundEstimationAdaptive import optimalAlphaAdaptive, optimalAlphaAdaptiveStabilized
import numpy as np

# Deliver 4. Video Surveillance for Road Traffic Monitoring

# Datasets
OF_dataset = 'Datasets/OpticalFlow/'
gt_path = 'Datasets/OpticalFlow/GT/'
pred_path = 'results/OpticalFlow/files'

# Task 1.1. Optical Flow with Block Matching

# Find optimal block size and area of search for Block Matching
MSEN_img1, MSEN_img2, PEPN_img1, PEPN_img2, blockSize_vec, areaOfSearch_vec = optimalBlockSizeArea(OF_dataset, gt_path, False)
np.save('task1/MSEN_img1', MSEN_img1)
np.save('task1/MSEN_img2', MSEN_img2)
np.save('task1/PEPN_img1', PEPN_img1)
np.save('task1/PEPN_img2', PEPN_img2)
MSEN_img1, PEPN_img1 = np.load('task1/MSEN_img1.npy'), np.load('task1/PEPN_img1.npy')
MSEN_img2, PEPN_img2 = np.load('task1/MSEN_img2.npy'), np.load('task1/PEPN_img2.npy')
blockSize_vec, areaOfSearch_vec = np.arange(20, 61, 10), np.arange(5, 51, 5)
plotMSENPEPNCurves(MSEN_img1, MSEN_img2, PEPN_img1, PEPN_img2, blockSize_vec, areaOfSearch_vec)

# Compute and evaluate Optical Flow with a specific block size and area of search
estimateOpticalFlow(OF_dataset, 40, 35, Backward=False, showResult=True)
evaluateOpticalFlow(pred_path, gt_path)

# Task 1.2. Block Matching vs Other Techniques

#Gunnar Farneback's method
estimateOpticalFlowFarneback(OF_dataset, Backward=False, showResult=True)
evaluateOpticalFlow(pred_path, gt_path)

# TV-L1 Optical Flow
estimateOpticalFlowTVL1(OF_dataset, Backward=False, showResult=True)
evaluateOpticalFlow(pred_path, gt_path)

# visualize ground-truth
plotOpticalFlowHSV(gt_path, visualization='HS')


# Task 2.1. Video Stabilization with Block Matching
sequence_path = 'Datasets/Stabilization/traffic/input/'
videoStabilizationPairs(sequence_path, blockSize=15, areaOfSearch=20, ID='traffic')
motionVectors, point = videoStabilizationReference(sequence_path, blockSize=30, areaOfSearch=10, Backward=False, ID='traffic')
motionVectors, point = videoStabilizationMovement(sequence_path, blockSize=30, areaOfSearch=10, Backward=False, ID='traffic')

dataset_path, ID3 = 'Datasets/Stabilization/', 'traffic'
precision_ID3_w3, recall_ID3_w3, fscore_ID3_w3, alpha_vec = optimalAlphaAdaptive(dataset_path, ID3, 0.2, True, True, 750, 4, True, False)
precision_ID3_w4, recall_ID3_w4, fscore_ID3_w4, alpha_vec = optimalAlphaAdaptiveStabilized(dataset_path, ID3, 0.2, True, True, 750, 4, True, False)
precisionRecallCurveDataset(precision_ID3_w3, precision_ID3_w4, recall_ID3_w3, recall_ID3_w4, ID3)

# Task 2.2. Block Matching stabilization vs Other Techniques

# Homography-based video stabilization
HomographyTransformStab(sequence_path)

# Task 2.3. Stabilize you own videos
src_path = 'Datasets/Stabilization/video/videoToStabilize.mp4'
dst_path = 'Datasets/Stabilization/video/frames/'
convertVideotoFrames(src_path, dst_path)
motionVectors = videoStabilizationMovement(dst_path, blockSize=30, areaOfSearch=10, Backward=False, ID='video')



