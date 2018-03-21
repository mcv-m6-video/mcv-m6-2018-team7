## Week 4

**Task 1.1.** Optical Flow with Block Matching (using MSE as matching cost).
  - Run the function optimalAlpha() and f1ScoreCurve() in main.py for F-score vs alpha. 
  - Run the function precisionRecallCurve() in main.py for AUC. 
  - See the implementation of optimalAlpha() in backgroundEstimation.py
  - See the implementation of f1ScoreCurve() and precisionRecallCurve() in readInputWriteOutput.py

**Task 1.2.** Block Matching vs other techniques (Farnebäck's method and TV-L1 Optical Flow). 
  - Run the function gridSearchAdaptiveClassifier() in main.py to find optimal parameters for the adaptive model.
  - The adaptive model is implemented in the class AdaptiveClassifier (check the associated .py file). 
  - Run the function optimalAlphaAdaptive() for F-score vs alpha (implemented in backgroundEstimationAdaptive.py).
  - Run the function precisionRecallCurve() in main.py for AUC.
  
**Task 2.1.** Video Stabilization with Block Matching (experiments with 2 approaches).
  - Run the function testBackgroundSubtractorMOG() in main.py
  - See the implementation of testBackgroundSubtractorMOG() in stateOfTheArt.py
  
**Task 2.2.** Block Matching Video Stabilization vs other techiques (Pyramidal Lucas-Kanade and Homography-based).
  - Set the boolean 'Color' to 'True' in main.py 
  - Run the same functions from Task 1, which are adapted to work with 3-channel images.
  - Uncomment line 55 of backgroundEstimation.py to use YCbCr. 

**Task 2.3.** Video Stabilization of videos of our own.
  - Set the boolean 'Color' to 'True' in main.py 
  - Run the same functions from Task 1, which are adapted to work with 3-channel images.
  - Uncomment line 55 of backgroundEstimation.py to use YCbCr. 
