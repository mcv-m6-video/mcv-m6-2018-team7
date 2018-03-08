from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from readInputWriteOutput import readDataset


def modelGaussianDistribution(train_frames):
    # Compute mean and std per frame
    mu_vec = np.mean(train_frames, 0)
    sigma_vec = np.std(train_frames, 0)

    return mu_vec, sigma_vec


def predictBackgroundForeground(test_frames, mu_vec, sigma_vec, alpha, Color):

    if Color == False:

        # Predict background and foreground with grayscale
        predictions = np.zeros([1, test_frames.shape[1]])

        for idx in range(len(test_frames)):
            current_frame = test_frames[idx, :]
            prediction = (abs(current_frame - mu_vec) >= alpha * (sigma_vec + 2)).astype(int)
            predictions = np.vstack((predictions, prediction))

    else:
        # Predict background and foreground with color
        predictions = np.zeros([1, test_frames.shape[1]])

        for idx in range(len(test_frames)):
            current_frame = test_frames[idx, :, :]
            prediction = (abs(current_frame - mu_vec) >= alpha * (sigma_vec + 2)).astype(int)
            prediction = np.all(prediction, 1).astype(int)
            predictions = np.vstack((predictions, prediction))

    predictions = predictions[1:, :]
    return predictions


def evaluateBackgroundEstimation(predictions, gts):

    predictions = np.reshape(predictions, gts.shape[0]*gts.shape[1])
    gts = np.reshape(gts, gts.shape[0] * gts.shape[1])

    #Remove array positions where gt >=85 & gt<=170
    predictions = predictions[gts != -1]
    gts = gts[gts != -1]

    precision, recall, fscore, support = precision_recall_fscore_support(gts, predictions, average = 'binary')
    #print 'F1-Score = ',fscore
    return precision, recall, fscore


def optimalAlpha(dataset_path, ID, Color):

    print 'Computing optimal alpha for '+ID+' dataset ...'

    Precision_vec = []
    Recall_vec = []
    fscore_vec = []
    alpha_vec = np.arange(0, 30, 0.2)

    train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID, Color)
    mu_vec, std_vec = modelGaussianDistribution(train_frames)

    ## To write the mean and std amage
    #mu = np.reshape(mu_vec, (240, 320, 3))
    #std = np.reshape(std_vec, (240, 320, 3))
    #cv2.imwrite("mu_Color_ID3.png", mu.astype(np.uint8))
    #cv2.imwrite("std_Color_ID3.png", std.astype(np.uint8))

    for alpha in alpha_vec:

        predictions = predictBackgroundForeground(test_frames, mu_vec, std_vec, alpha, Color)
        precision, recall, fscore = evaluateBackgroundEstimation(predictions, test_gts)

        Precision_vec = np.append(Precision_vec, precision)
        Recall_vec = np.append(Recall_vec, recall)
        fscore_vec = np.append(fscore_vec, fscore)

    min, max, idxmin, idxmax = cv2.minMaxLoc(fscore_vec)
    print 'Maximum F1-Score with ', ID, ' dataset is ', max, ' with alpha = ', alpha_vec[idxmax[1]]
    print 'Precision selected with dataset ', ID, ' is ', Precision_vec[idxmax[1]]
    print 'Recall selected with dataset ', ID, ' is ', Recall_vec[idxmax[1]]

    # Plot F1, Precision and recall per alpha
    fig = plt.figure()
    fig.suptitle('F-Score, Precision and Recall vs alpha (' + ID + ')', fontsize=15)
    ax3 = fig.add_subplot(111)
    ax3.set_xlabel('Alpha', fontsize=11)
    ax3.plot(alpha_vec, fscore_vec, c='orange', linewidth=3.0, label='F1-Score')
    ax3.plot(alpha_vec, Recall_vec, c='g', linewidth=3.0, label='Recall')
    ax3.plot(alpha_vec, Precision_vec, c='b', linewidth=3.0, label='Precision')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotPrecRecFsc_' + ID + '.png', bbox_inches='tight')

    return Precision_vec, Recall_vec, fscore_vec, alpha_vec


def testBackgroundEstimation(alpha, dataset_path, ID, Color):
    train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID, Color)
    mu_vec, std_vec = modelGaussianDistribution(train_frames)
    predictions = predictBackgroundForeground(test_frames, mu_vec, std_vec, alpha, Color)
    precision, recall, fscore = evaluateBackgroundEstimation(predictions, test_gts)
    print 'F-score:', fscore
    print 'Precision: ', precision
    print 'Recall: ', recall
