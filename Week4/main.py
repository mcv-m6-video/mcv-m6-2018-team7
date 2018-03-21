from opticalFlowEstimation import estimateOpticalFlow, optimalBlockSizeArea, estimateOpticalFlowFarneback, estimateOpticalFlowTVL1
from opticalFlowEvaluation import evaluateOpticalFlow
from readInputWriteOutput import plotMSENPEPNCurves, plotOpticalFlowHSV, convertVideotoFrames
from videoStabilization import videoStabilizationReference, videoStabilizationMovement, videoStabilizationPairs
import numpy as np
# Deliver 4. Video Surveillance for Road Traffic Monitoring

# Datasets
OF_dataset = 'Datasets/OpticalFlow/'
gt_path = 'Datasets/OpticalFlow/GT/'
pred_path = 'results/OpticalFlow/files'

# Task 1.1. Optical Flow with Block Matching
# estimateOpticalFlow(OF_dataset, blockSize = 50, areaOfSearch = 8, Backward = False, showResult = True)
# evaluateOpticalFlow(pred_path, gt_path)    # Per fer evaluation showResults ha de estar a False
#MSEN_img1, MSEN_img2, PEPN_img1, PEPN_img2, blockSize_vec, areaOfSearch_vec = optimalBlockSizeArea(OF_dataset, gt_path, False)
#np.save('task1/MSEN_img1', MSEN_img1)
#np.save('task1/MSEN_img2', MSEN_img2)
#np.save('task1/PEPN_img1', PEPN_img1)
#np.save('task1/PEPN_img2', PEPN_img2)
MSEN_img1 = np.load('task1/MSEN_img1.npy')
MSEN_img2 = np.load('task1/MSEN_img2.npy')
PEPN_img1 = np.load('task1/PEPN_img1.npy')
PEPN_img2 = np.load('task1/PEPN_img2.npy')
blockSize_vec = np.arange(20, 61, 10)
areaOfSearch_vec = np.arange(5, 51, 5)
#plotMSENPEPNCurves(MSEN_img1, MSEN_img2, PEPN_img1, PEPN_img2, blockSize_vec, areaOfSearch_vec)

estimateOpticalFlow(OF_dataset, 40, 35, Backward=False, showResult=True)
evaluateOpticalFlow(pred_path, gt_path)

# Task 1.2. Block Matching vs Other Techniques
#estimateOpticalFlowFarneback(OF_dataset, Backward=False, showResult=True)
#evaluateOpticalFlow(pred_path, gt_path)
estimateOpticalFlowTVL1(OF_dataset, Backward=False, showResult=True)
evaluateOpticalFlow(pred_path, gt_path)
plotOpticalFlowHSV(gt_path, visualization='HS')


# Task 2.1. Video Stabilization with Block Matching
sequence_path = 'Datasets/Stabilization/traffic/input/'
videoStabilizationPairs(sequence_path, blockSize=15, areaOfSearch=20)
motionVectors, point = videoStabilizationReference(sequence_path, blockSize= 30, areaOfSearch = 10 , Backward = False)
motionVectors, point = videoStabilizationMovement(sequence_path, blockSize= 30, areaOfSearch = 10 , Backward = False)

# Task 2.2. Block Matching stabilization vs Other Techniques



# Task 2.3. Stabilize you own videos
src_path = 'Datasets/Stabilization/Video/videoToStabilize.mp4'
dst_path = 'Datasets/Stabilization/Video/frames/'
convertVideotoFrames(src_path, dst_path)
motionVectors = videoStabilizationMovement(dst_path, blockSize= 30, areaOfSearch = 10, Backward = False)



