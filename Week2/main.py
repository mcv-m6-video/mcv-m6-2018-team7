from backgroundEstimation import testBackgroundEstimation, optimalAlpha
from readInputWriteOutput import f1ScoreCurve, precisionRecallCurve
from backgroundEstimationAdaptive import testAdaptiveClassifier, gridSearchAdaptiveClassifier, optimalAlphaAdaptive
from stateOfTheArt import testBackgroundSubtractorMOG

# Deliver 2. Video Surveillance for Road Traffic Monitoring

ID1 = 'highway'
ID2 = 'fall'
ID3 = 'traffic'
dataset_path = 'Datasets/'
Color = False

# Task 1.1 Gaussian modelling for background-foreground estimation
testBackgroundEstimation(1.8, dataset_path, ID1, Color)

# Task 1.2 & 1.3 Evaluate results
precision_ID1, recall_ID1, fscore_ID1, alpha_vec = optimalAlpha(dataset_path, ID1, Color)
precision_ID2, recall_ID2, fscore_ID2, alpha_vec = optimalAlpha(dataset_path, ID2, Color)
precision_ID3, recall_ID3, fscore_ID3, alpha_vec = optimalAlpha(dataset_path, ID3, Color)
f1ScoreCurve(fscore_ID1, fscore_ID2, fscore_ID3, alpha_vec)
precisionRecallCurve(precision_ID1, precision_ID2, precision_ID3, recall_ID1, recall_ID2, recall_ID3)

# Task 2.1 Recursive Gaussian modeling
testAdaptiveClassifier(1.8, 0., dataset_path, ID1)
#testAdaptiveClassifier(1.8, 0.2, dataset_path, ID1)

# Task 2.2 Evaluate and compare to non-recursive
gridSearchAdaptiveClassifier(dataset_path, ID3, Color)
precision_ID1, recall_ID1, fscore_ID1, alpha_vec = optimalAlphaAdaptive(dataset_path, ID1, Color, 0.2)
precision_ID2, recall_ID2, fscore_ID2, alpha_vec = optimalAlphaAdaptive(dataset_path, ID2, Color, 0.05)
precision_ID3, recall_ID3, fscore_ID3, alpha_vec = optimalAlphaAdaptive(dataset_path, ID3, Color, 0.2)
f1ScoreCurve(fscore_ID1, fscore_ID2, fscore_ID3, alpha_vec)
precisionRecallCurve(precision_ID1, precision_ID2, precision_ID3, recall_ID1, recall_ID2, recall_ID3)

# Task 3 Comparison with state of the art
testBackgroundSubtractorMOG(dataset_path, ID1, 'MOG2', True, Color)

# Task 4 Color sequences
Color = True
testBackgroundEstimation(1.8, dataset_path, ID1, Color)

# Task 4.2  Color sequences Evaluation
precision_ID1, recall_ID1, fscore_ID1, alpha_vec = optimalAlpha(dataset_path, ID1, Color)
precision_ID2, recall_ID2, fscore_ID2, alpha_vec = optimalAlpha(dataset_path, ID2, Color)
precision_ID3, recall_ID3, fscore_ID3, alpha_vec = optimalAlpha(dataset_path, ID3, Color)
f1ScoreCurve(fscore_ID1, fscore_ID2, fscore_ID3, alpha_vec)
precisionRecallCurve(precision_ID1, precision_ID2, precision_ID3, recall_ID1, recall_ID2, recall_ID3)






