# Video Surveillance for Road Traffic Monitoring
Master in Computer Vision, Barcelona (2017-2018) - M6 Video Analysis

## About us
We are Team 7:   
[Roger Marí](https://github.com/rogermm14). Email: roger.mari01@estudiant.upf.edu  
[Joan Sintes](https://github.com/JoSintes8). Email: joan.sintes01@estudiant.upf.edu  
[Àlex Palomo](https://github.com/alexpalomodominguez). Email: alex.palomo01@estudiant.upf.edu  
[Àlex Vicente](https://github.com/AlexVicenteS). Email: alex.vicente01@estudiant.upf.edu  

## Abstract
This 5-week project presents a series of experiments related to video analysis for traffic monitoring.   
We will employ basic concepts and techniques related to video sequences mainly for surveillance applications,     
divided in 4 main stages: background estimation, foreground segmentation, video stabilization and region tracking. 

## Week 1. Metrics and tools for Background Subtraction / OF evaluation.
**Task 1.** Background substraction. Segmentation metrics. Precision and recall.   
**Task 2.** Background susbtraction. Segmentation metrics. Temporal analysis.   
**Task 3.** Optical flow evaluation metrics. Mean Squared Error and Percentage of Erroneous Pixels in Non-occluded areas.   
**Task 4.** Background substraction. Evaluation of de-synchornized results.   
**Task 5.** Visual representation of optical flow.    
- [x] Check the information about how to run the code [here](https://github.com/mcv-m6-video/mcv-m6-2018-team7/blob/master/Week1/README.md)

## Week 2. Background Subtraction via Gaussian Modelling.
**Task 1.** Gaussian modelling. Evaluation by means of F-score vs alpha and AUC (Precision-Recall curves).   
**Task 2.** Adaptive modelling. Comparison between adaptive and non-adaptive methods via F-score and AUC.  
**Task 3.** Comparison with state of the art. Methods from Tasks 1 and 2 are compared to MOG and MOG2.   
**Task 4.** Gaussian modelling taking into account color. RGB and YCbCr colorspaces used.    
- [x] Check the information about how to run the code [here](https://github.com/mcv-m6-video/mcv-m6-2018-team7/blob/master/Week2/README.md)

## Week 3. Post-processing techniques for Background Subtraction.
**Task 1.** Hole filling to complete objects in the foreground.   
**Task 2.** Area filtering to remove noise from the background.    
**Task 3.** Morphological operators (closing + hole filling) to boost perfromance.        
**Task 4.** Shadow detection and removal (pixel based methods using the HSV colorspace).    
**Task 5.** Improvement in Precision-Recall curves with respect to the best configuration from week 2.    
- [x] Check the information about how to run the code [here](https://github.com/mcv-m6-video/mcv-m6-2018-team7/blob/master/Week3/README.md)

## Week 4. Optical Flow with Block Matching and Video Stabilization.
**Task 1.1.** Optical Flow with Block Matching (using MSE as matching cost).    
**Task 1.2.** Block Matching vs other techniques (Farnebäck's method and TV-L1 Optical Flow).     
**Task 2.1.** Video Stabilization with Block Matching (experiments with 2 approaches).     
**Task 2.2.** Block Matching Video Stabilization vs other techiques (Pyramidal Lucas-Kanade and Homography-based).     
**Task 2.3.** Video Stabilization of videos of our own.     
- [x] Check the information about how to run the code [here](https://github.com/mcv-m6-video/mcv-m6-2018-team7/blob/master/Week4/README.md)

## Week 5. Vehicle Tracking and Speed Estimation.   
**Task 1.1.** Tracking with Kalman Filter (and bounding box merging).    
**Task 1.2.** Other tracking methods. Median-Flow tracker.    
**Task 2.** Speed estimation via homography rectification.   
**Task 3.** Own study: car density (cars/frame), traffic rate (cars/minute) and infraction detection (speed limit 80km/h).   
- [x] Check the information about how to run the code [here](https://github.com/mcv-m6-video/mcv-m6-2018-team7/blob/master/Week5/README.md)    
