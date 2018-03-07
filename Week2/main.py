from backgroundEstimation import modelGaussianDistribution, readDataset, predictBackgroundForeground
from evaluation import evaluateBackgroudEstimation, optimalAlpha, precisionRecallCurve, f1ScoreCurve
from AdaptiveClassifier import AdaptiveClassifier
from backgroundEstimation import gridSearchAdaptiveClassifier
from stateOfTheArt import backgroundSubtractorMOG
import numpy as np
# Deliver 2. Video Surveillance for Road

ID1 = 'highway'
ID2 = 'fall'
ID3 = 'traffic'
dataset_path = 'Datasets/'

# use color version?
Color = False

# Task 1.1 Gaussian distribution
train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID1, Color)
mu_vec, std_vec = modelGaussianDistribution(train_frames)
predictions = predictBackgroundForeground(test_frames, mu_vec, std_vec, 1.8, Color)
precision, recall, fscore = evaluateBackgroudEstimation(predictions, test_gts)
print fscore

# Task 1.2 & 1.3 Evaluate results
precision_ID1, recall_ID1, fscore_ID1, alpha_vec = optimalAlpha(dataset_path, ID1, Color)
precision_ID2, recall_ID2, fscore_ID2, alpha_vec = optimalAlpha(dataset_path, ID2, Color)
precision_ID3, recall_ID3, fscore_ID3, alpha_vec = optimalAlpha(dataset_path, ID3, Color)
f1ScoreCurve(fscore_ID1, fscore_ID2, fscore_ID3, alpha_vec)
precisionRecallCurve(precision_ID1, precision_ID2, precision_ID3, recall_ID1, recall_ID2, recall_ID3)

# Task 2.1 Recursive Gaussian modeling
# Task 2.2 Evaluate and compare to non-recursive
train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID1, Color)
init_mu = np.zeros([1, train_frames.shape[1]])
init_std = np.zeros([1, train_frames.shape[1]])
clf = AdaptiveClassifier(1.8, 0., init_mu, init_std)
clf = clf.fit(train_frames,train_gts)
fscore = clf.score(test_frames,test_gts)
print fscore

# Grid search to optimize parameters rho and alpha
#alpha_opt, rho_opt, fscore = gridSearchAdaptiveClassifier(dataset_path, ID3, Color)

# Task 3 Comparison with state of the art
method = 'MOG2'
ID=ID2
train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID, Color)
predictions_sota = backgroundSubtractorMOG(dataset_path, ID, method, test_frames)
precision, recall, fscore = evaluateBackgroudEstimation(predictions_sota, test_gts)
print ('fscore:'+str(fscore)+ ' recall'+str(recall)+'precision'+str(precision))

# Task 4 Color sequences
Color = True
train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID1, Color)
mu_vec, std_vec = modelGaussianDistribution(train_frames)
predictions = predictBackgroundForeground(test_frames, mu_vec, std_vec, 1.8, Color)
precision, recall, fscore = evaluateBackgroudEstimation(predictions, test_gts)
print fscore







