## Week 5. Vehicle Tracking and Speed Estimation.   

**Task 1.1.** Tracking with Kalman Filter (and bounding box merging).   
  - Run the function testKalmanTracking() in main.py with speed=False to perform only tracking with Kalman filter.    
  - Tracking with Kalman filter is implmenented in trackingKalman.py. Check the filter paramters in KalmanFilter.py.   
  - The function mergeCloseBoundingBoxes() is used to enable certain cars to have more than one connected component.      

**Task 1.2.** Tracking with other tracking methods. The Median-Flow tracker.   
  - Run the function testMedianFlowTracking() in main.py to perform tracking with OpenCV's Median-Flow tracker.   
  - OpenCV version 3.0 or higher required.   
  - We added the functions in utilsTracking.py to initialize the tracker for each vehicle and for bbox merging.   
  
**Task 2.** Speed estimation via homography rectification.  
  - Run the function testKalmanTracking() in main.py with speed=True to perform tracking with Kalman plus speed estimation.  
  - All necessary tools for speed estimation with our method are implemented in estimateSpeed.py.   
  - The speed of each vehicle is periodically updated every 3 fames (see the update method in ObjectDetected.py).   
  
**Task 3.** Own study: car density (cars/frame), traffic rate (cars/minute) and infraction detection (speed limit 80km/h).   
  - Run the function testKalmanTracking() in main.py with speed=True and our study statistics will be displayed.    
  - Our study statistics are computed in paintTrackingResults() and displayed via drawStatistics (see trackingKalman.py).   
  
