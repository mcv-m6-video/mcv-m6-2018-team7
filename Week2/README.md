## Week 2

**Task 1.** Gaussian modelling. Evaluation by means of F-score vs alpha and AUC (Precision-Recall curves).      
  - Run the function optimalAlpha() and f1ScoreCurve in main.py for F-score vs alpha    
  - Run the function precisionRecallCurve() in main.py for AUC
  - See the implementation of optimalAlpha(), f1ScoreCurve() and precisionRecallCurve() in evaluation.py

**Task 2.** Adaptive modelling. Comparison between adaptive and non-adaptive methods via F-score and AUC. 
  - Run the function gridSearchAdaptiveClassifier() in main.py
  - The adaptive model is implemented in the class AdaptiveClassifier (check the associated .py file).  
  
**Task 3.** Comparison with state of the art. Methods from Tasks 1 and 2 are compared to MOG and MOG2.
  - Run the functions readDataset(), backgroundSubstractionMOG() and evaluateBackgroundEstimaton() in main.py
  - See the implementation of readDataset() in backgroundEstimation.py
  - See the implementation of backgroundSubstractionMOG() in stateOfTheArt.py.
  - See the implementation of evaluateBackgroundEstimation() in evaluation.py
  
**Task 4.** Gaussian modelling taking into account color. RGB and YCbCr colorspaces used.
  - Set the boolean 'Color' to 'True' in main.py 
  - Run the same functions from Task 1, they are adapted to deal with 3-channel images.
  - Uncomment line 55 of backgroundEstimation.py to use YCbCr. 
