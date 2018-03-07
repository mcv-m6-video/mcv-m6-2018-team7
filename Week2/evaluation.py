from __future__ import division
import os
import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, auc
import matplotlib.pyplot as plt
from backgroundEstimation import *


def evaluateBackgroudEstimation(predictions, gts):

    predictions = np.reshape(predictions, predictions.shape[0]*predictions.shape[1])
    gts = np.reshape(gts, gts.shape[0] * gts.shape[1])

    #Remove array positions where gt >=85 & gt<=170
    predictions = predictions[gts != -1]
    gts = gts[gts != -1]

    precision, recall, fscore, support = precision_recall_fscore_support(gts, predictions, average = 'binary')
    #print 'F1-Score = ',fscore
    return precision, recall, fscore


def optimalAlpha(dataset_path, ID, Color):

    Precision_vec = []
    Recall_vec = []
    fscore_vec = []
    alpha_vec = np.arange(0, 10, 0.25)

    train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID, Color)
    mu_vec, std_vec = modelGaussianDistribution(train_frames)

    ## To write the mean and std amage
    #mu = np.reshape(mu_vec, (240, 320, 3))
    #std = np.reshape(std_vec, (240, 320, 3))
    #cv2.imwrite("mu_Color_ID3.png", mu.astype(np.uint8))
    #cv2.imwrite("std_Color_ID3.png", std.astype(np.uint8))

    for alpha in alpha_vec:

        predictions = predictBackgroundForeground(test_frames, mu_vec, std_vec, alpha, Color)
        precision, recall, fscore = evaluateBackgroudEstimation(predictions, test_gts)

        Precision_vec = np.append(Precision_vec, precision)
        Recall_vec = np.append(Recall_vec, recall)
        fscore_vec = np.append(fscore_vec, fscore)

    min, max, idxmin, idxmax = cv2.minMaxLoc(fscore_vec)
    print 'Maximum F1-Score with dataset ', ID, ' is ', max, ' with alpha = ', alpha_vec[idxmax[1]]

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


def precisionRecallCurve(precision_ID1, precision_ID2, precision_ID3, recall_ID1, recall_ID2, recall_ID3):


    AUC_ID1 = auc(recall_ID1, precision_ID1)
    AUC_ID2 = auc(recall_ID2, precision_ID2)
    AUC_ID3 = auc(recall_ID3, precision_ID3)

    # Plot Precision-Recall Curve
    fig = plt.figure()
    fig.suptitle('Precision-Recall Curve', fontsize=15)
    ax3 = fig.add_subplot(111)
    ax3.set_xlabel('Recall', fontsize=11)
    ax3.set_ylabel('Precision', fontsize=11)
    ax3.set_xlim([0,1])
    ax3.set_ylim([0,1])
    ax3.plot(recall_ID1, precision_ID1, c='orange', linewidth=3.0, label ='highway')
    ax3.plot(recall_ID2, precision_ID2, c='g', linewidth=3.0, label='fall')
    ax3.plot(recall_ID3, precision_ID3, c='b', linewidth=3.0, label='traffic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotPrecisionRecallCurve.png', bbox_inches='tight')

    print 'Area Under Curve (AUC) of dataset ID1 = ', AUC_ID1
    print 'Area Under Curve (AUC) of dataset ID2 = ', AUC_ID2
    print 'Area Under Curve (AUC) of dataset ID3 = ', AUC_ID3

    return


def f1ScoreCurve(fscore_ID1, fscore_ID2, fscore_ID3,alpha_vec):

    # Plot Precision-Recall Curve
    fig = plt.figure()
    fig.suptitle('F1-Score vs Alpha', fontsize=15)
    ax3 = fig.add_subplot(111)
    ax3.set_xlabel('alpha', fontsize=11)
    ax3.set_ylabel('F1-Score', fontsize=11)
    ax3.set_xlim([0, np.max(alpha_vec)])
    ax3.set_ylim([0, 1])
    ax3.plot(alpha_vec, fscore_ID1, c='orange', linewidth=3.0, label ='highway')
    ax3.plot(alpha_vec, fscore_ID2, c='g', linewidth=3.0, label='fall')
    ax3.plot(alpha_vec, fscore_ID3, c='b', linewidth=3.0, label='traffic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotF1-ScoreCurve.png', bbox_inches='tight')

    return
