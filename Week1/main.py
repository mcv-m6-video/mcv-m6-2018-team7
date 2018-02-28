from evaluation import readFolder, evaluateFolder, evaluateTemporal, dSyncGlobalEvaluation, dSyncTemporalEvaluation
from optical_flow import evaluateOpticalFlow, plotOpticalFlowHSV

# Deliver 1. Video Surveillance for Road

# Task 1. Background substraction (quantitative evaluation)
pred_dir = 'test_results_foreground/results/highway/A'
gt_dir = 'highway/groundtruth'
pred_vec, gt_vec = readFolder(pred_dir, gt_dir)
#evaluateFolder(pred_vec, gt_vec)

# Task 2. Background substraction (temporal evaluation)
#evaluateTemporal(pred_dir, gt_dir)

# Task 3. Optical flow evaluation
pred_OF_dir = 'test_results_motion/results'
gt_OF_dir = 'kitti_optical_flow'
#evaluateOpticalFlow(pred_OF_dir,gt_OF_dir)

# Task 4. De-synchornization of results
pred_dirA = 'test_results_foreground/results/highway/A'
pred_dirB = 'test_results_foreground/results/highway/B'
pred_vecA, gt_vec = readFolder(pred_dirA, gt_dir)
pred_vecB, gt_vec = readFolder(pred_dirB, gt_dir)
#dSyncGlobalEvaluation(pred_vecA, pred_vecB, gt_vec)

pred_dir = 'test_results_foreground/results/highway/A'
#dSyncTemporalEvaluation(pred_dir, gt_dir)

# Task 5. Optical flow visualization
plotOpticalFlowHSV(gt_OF_dir)