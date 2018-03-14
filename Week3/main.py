from backgroundEstimation import testBackgroundEstimation, optimalAlpha
from readInputWriteOutput import f1ScoreCurve, precisionRecallCurve, aucCurve, precisionRecallCurveDataset
from backgroundEstimationAdaptive import testAdaptiveClassifier, gridSearchAdaptiveClassifier, optimalAlphaAdaptive, optimalPAdaptive
import pickle
import numpy as np

# Deliver 3. Video Surveillance for Road Traffic Monitoring

##################### BEST CONFIGURATIONS WEEK 2 ###############################
#                                                                              #
#  Highway -> F1 = 0.723 AUC = 0.683 (Adaptive with alpha = 2.5 and rho = 0.2) #
#                                                                              #
#  Fall    -> F1 = 0.695 AUC = 0.702 (Adaptive with alpha = 3 and rho = 0.05)  #
#                                                                              #
#  Traffic -> F1 = 0.659 AUC = 0.615 (Adaptive with alpha = 3.5 and rho = 0.2) #
#                                                                              #
################################################################################

# Datasets
ID1 = 'highway'
ID2 = 'fall'
ID3 = 'traffic'
dataset_path = 'Datasets/'

# Set to True those tasks that you want to run
task1, task2, task3, task4, task5 = False, True, False, False, False
connectivity = 4

# Task 1. Hole Filling
if task1:
    holeFilling, areaFiltering, Morph, shadRemov = True, False, False, False
    precision_ID1, recall_ID1, fscore_ID1, alpha_vec = optimalAlphaAdaptive(dataset_path, ID1, 0.2, holeFilling, areaFiltering, 50, connectivity, Morph, shadRemov)
    precision_ID2, recall_ID2, fscore_ID2, alpha_vec = optimalAlphaAdaptive(dataset_path, ID2, 0.05, holeFilling, areaFiltering, 500, connectivity, Morph, shadRemov)
    precision_ID3, recall_ID3, fscore_ID3, alpha_vec = optimalAlphaAdaptive(dataset_path, ID3, 0.2, holeFilling, areaFiltering, 750, connectivity, Morph, shadRemov)
    f1ScoreCurve(fscore_ID1, fscore_ID2, fscore_ID3, alpha_vec)
    precisionRecallCurve(precision_ID1, precision_ID2, precision_ID3, recall_ID1, recall_ID2, recall_ID3)

# Task 2.1. and 2.2. Area filtering AUC vs Pixels / Area filtering argmax P (AUC)
if task2:
    holeFilling, areaFiltering, Morph, shadRemov = True, True, False, False
    auc_vec_ID1, p_vec = optimalPAdaptive(dataset_path, ID1, 0.2, holeFilling, areaFiltering, connectivity)
    auc_vec_ID2, p_vec = optimalPAdaptive(dataset_path, ID2, 0.05, holeFilling, areaFiltering, connectivity)
    auc_vec_ID3, p_vec = optimalPAdaptive(dataset_path, ID3, 0.2, holeFilling, areaFiltering, connectivity)
    aucCurve(auc_vec_ID1, auc_vec_ID2, auc_vec_ID3, p_vec)

# Task 3. Additional Morphological processings
if task3:
    holeFilling, areaFiltering, Morph, shadRemov = True, True, True, False
    precision_ID1, recall_ID1, fscore_ID1, alpha_vec = optimalAlphaAdaptive(dataset_path, ID1, 0.2, holeFilling, areaFiltering, 50, connectivity, Morph, shadRemov)
    precision_ID2, recall_ID2, fscore_ID2, alpha_vec = optimalAlphaAdaptive(dataset_path, ID2, 0.05, holeFilling, areaFiltering, 500, connectivity, Morph, shadRemov)
    precision_ID3, recall_ID3, fscore_ID3, alpha_vec = optimalAlphaAdaptive(dataset_path, ID3, 0.2, holeFilling, areaFiltering, 750, connectivity, Morph, shadRemov)
    f1ScoreCurve(fscore_ID1, fscore_ID2, fscore_ID3, alpha_vec)
    precisionRecallCurve(precision_ID1, precision_ID2, precision_ID3, recall_ID1, recall_ID2, recall_ID3)

# Task 4. Shadow Removal
if task4:
    holeFilling, areaFiltering, Morph, shadRemov = True, True, True, True
    precision_ID1, recall_ID1, fscore_ID1, alpha_vec = optimalAlphaAdaptive(dataset_path, ID1, 0.2, holeFilling, areaFiltering, 50, connectivity, Morph, shadRemov)
    precision_ID2, recall_ID2, fscore_ID2, alpha_vec = optimalAlphaAdaptive(dataset_path, ID2, 0.05, holeFilling, areaFiltering, 500, connectivity, Morph, shadRemov)
    precision_ID3, recall_ID3, fscore_ID3, alpha_vec = optimalAlphaAdaptive(dataset_path, ID3, 0.2, holeFilling, areaFiltering, 750, connectivity, Morph, shadRemov)
    f1ScoreCurve(fscore_ID1, fscore_ID2, fscore_ID3, alpha_vec)
    precisionRecallCurve(precision_ID1, precision_ID2, precision_ID3, recall_ID1, recall_ID2, recall_ID3)

# Task 5. Improvements this week
if task5:
    #Highway Dataset
    precision_ID1_w2, recall_ID1_w2, fscore_ID1_w2, alpha_vec = optimalAlphaAdaptive(dataset_path, ID1, 0.2, False, False, 50, connectivity, False, False)
    precision_ID1_w3, recall_ID1_w3, fscore_ID1_w3, alpha_vec = optimalAlphaAdaptive(dataset_path, ID1, 0.2, True, True, 50, connectivity, True, False)
    precisionRecallCurveDataset(precision_ID1_w2, precision_ID1_w3, recall_ID1_w2, recall_ID1_w3, ID1)
    # #Fall Dataset
    precision_ID2_w2, recall_ID2_w2, fscore_ID2_w2, alpha_vec = optimalAlphaAdaptive(dataset_path, ID2, 0.05, False, False, 500, connectivity, False, False)
    precision_ID2_w3, recall_ID2_w3, fscore_ID2_w3, alpha_vec = optimalAlphaAdaptive(dataset_path, ID2, 0.05, True, True, 500, connectivity, True, False)
    precisionRecallCurveDataset(precision_ID2_w2, precision_ID2_w3, recall_ID2_w2, recall_ID2_w3, ID2)
    # #Traffic Dataset
    precision_ID3_w2, recall_ID3_w2, fscore_ID3_w2, alpha_vec = optimalAlphaAdaptive(dataset_path, ID3, 0.2, False, False, 750, connectivity, False, False)
    precision_ID3_w3, recall_ID3_w3, fscore_ID3_w3, alpha_vec = optimalAlphaAdaptive(dataset_path, ID3, 0.2, True, True, 750, connectivity, True, False)
    precisionRecallCurveDataset(precision_ID3_w2, precision_ID3_w3, recall_ID3_w2, recall_ID2_w3, ID3)


# To test the Foreground-Background classifier at a specific working point:
#holeFilling, areaFiltering, Morph, shadRemov = True, True, True, True
#testAdaptiveClassifier(dataset_path, ID1, 1.8, 0.2, holeFilling, areaFiltering, 50, connectivity, Morph, shadRemov)



