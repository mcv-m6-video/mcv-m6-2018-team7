## Week 4

**Task 1.1.** Optical Flow with Block Matching (using MSE as matching cost).   
  - Run the function optimalBlockSizeArea() in main.py to find best block size and area of search.    
  - Function plotMSENPEPNCurves() plots evolution of MSEN and PEPN for each block size and area of search.   
  - After parameter tuning, estimateOpticalFlow() is used to compute OF with the optimal block size and area of search.   

**Task 1.2.** Block Matching vs other techniques (Farnebäck's method and TV-L1 Optical Flow).    
  - Run the function estimateOpticalFlowFarneback() in main.py to compute OF using Farneback's method.   
  - Run the function estimateOpticalFlowTVL1() in main.py to compute OF using TV-L1 energy minimization.   
  - After each method, evaluateOpticalFlow() is used to compute the error of the estimated flow (MSEN and PEPN).   
  - Additionally, function plotOpticalFlowHSV() can be used to visualize the groud-truth flow.   
  
**Task 2.1.** Video Stabilization with Block Matching (experiments with 2 approaches).   
  - Run the function videoStabilizationPairs() in main.py.   
  - Use videoStabilizationReference() or videoStabilizationMovement() to use our approach 1 or 2 for video stabilization. 
  - Finally use optimalAlphaAdaptive(), optimalAlphaAdaptiveStabilized() and precisionRecallCurveDataset().       
    to plot Precision-Recall curves and get AUC of background subtraction before and after video stabilization.       
  
**Task 2.2.** Block Matching Video Stabilization vs other techiques (Pyramidal Lucas-Kanade and Homography-based).   
  - Run HomographyTransformStab() in main.py for homography based video stabilization.    
  - Run pyrLK_stabilization/video_stabilization.py for video stabilization based on pyramidal Lucas-Kanade OF estimation.    

**Task 2.3.** Video Stabilization of videos of our own.    
  - Complete the source and destination paths (where to read input video and write output stabilized video) in main.py.    
  - Function convertVideotoFrames() is used to extract the frames from the input video (we used mp4 format).   
  - Run videoStabilizationReference() or videoStabilizationMovement() to use our approach 1 or 2 for video stabilization.    
