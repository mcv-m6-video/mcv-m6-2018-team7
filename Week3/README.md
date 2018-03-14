## Week 3. Post-processing techniques for background subtraction

**Task 1.** Hole filling to complete objects in the foreground.        
  - Set the boolean 'task1' to True in the main.py and run.  
  - Set 'holeFilling' to True and the rest to False in the postprocessing booleans.   
  - backgroundEstimationAdaptive.optimalAlphaAdaptive() is executed for each dataset to find the AUC.   

**Task 2.** Area filtering to remove noise from the background.    
  - Set the boolean 'task2' to True in the main.py and run.   
  - Set 'holeFilling' and 'areaFiltering' to True and the rest to False in the postprocessing booleans.   
  - backgroundEstimationAdaptive.optimalPAdaptive() is executed to find the optimal P for each sequence.   
  
**Task 3.** Morphological operators (closing + hole filling) to boost perfromance.    
  - Set the boolean 'task3' to True in the main.py and run.   
  - Set 'holeFilling', 'areaFiltering' and 'Morph' to True and the rest to False in the postprocessing booleans.   
  - backgroundEstimationAdaptive.optimalAlphaAdaptive() is executed for each dataset to find the AUC.   
  - All post-processing techniques up to task 3 are implemented in AdaptiveClassifier.postProcessing().   
  
**Task 4.** Shadow detection and removal (pixel based methods using the HSV colorspace).   
  - Set the boolean 'task4' to True in the main.py and run.    
  - Set 'holeFilling', 'areaFiltering', 'Morph' and 'shadRemov' to True in the postprocessing booleans.    
  - Shadow removal is implemented in AdaptiveClassifier.shadowRemoval().   
  - To use method 1: uncomment lines 99-103 and comment lines 110-116.   
  - To use method 2: comment lines 99-103 and uncomment lines 110-116.    

**Task 4.** Improvement in Precision-Recall curves with respect to the best configuration from week 2.    
  - Set the boolean 'task5' to True in the main.py and run.   
  - readInputWriteOutput.precisionRecallCurveDataset() is used to plot the PR curves of each dataset.   
