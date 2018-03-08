from __future__ import division
import numpy as np
import cv2
import os
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def readDataset(dataset_path, ID, Color):
    # Get the ID dataset path
    frames_path = dataset_path + ID + '/input'
    gt_path = dataset_path + ID + '/groundtruth'

    # List all the files in the ID dataset path
    frame_list = sorted(os.listdir(frames_path))
    gt_list = sorted(os.listdir(gt_path))

    if Color == False:
        # Compute mean and sigma  of each pixel
        for idx in range(len(frame_list)):

            # print 'Evaluating frame ' + pr_name
            frame_dir = os.path.join(frames_path, frame_list[idx])
            gt_dir = os.path.join(gt_path, gt_list[idx])

            # Get frame
            frame = cv2.imread(frame_dir, 0)
            frame_vec = frame.ravel()

            # Get groundtruth with three different values (0 background, 1 foreground, -1 unknown)
            gt = cv2.imread(gt_dir, 0)
            gt_vec = gt.ravel()
            gt_vec[gt_vec <= 50] = 0
            gt_vec[(gt_vec >= 85) & (gt_vec <= 170)] = -1
            gt_vec[gt_vec > 170] = 1

            if idx == 0:
                frames = np.vstack((np.zeros([1, frame.shape[0] * frame.shape[1]]), frame_vec))
                gts = np.vstack((np.zeros([1, frame.shape[0] * frame.shape[1]]), gt_vec))
            else:
                frames = np.vstack((frames, frame_vec))
                gts = np.vstack((gts, gt_vec))
        frames = frames[1:, :]
        train_frames = frames[:int(round(frames.shape[0] * 0.5)), :]
        test_frames = frames[int(round(frames.shape[0] * 0.5)):, :]

    else:
        # Compute mean and sigma  of each pixel
        for idx in range(len(frame_list)):

            # print 'Evaluating frame ' + pr_name
            frame_dir = os.path.join(frames_path, frame_list[idx])
            gt_dir = os.path.join(gt_path, gt_list[idx])

            # Get frame
            frame = cv2.imread(frame_dir, 1)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB) #uncomment this line to use YCbCr

            frame_vec = np.reshape(frame, (1, frame.shape[0] * frame.shape[1], 3))

            # Get groundtruth with three different values (0 background, 1 foreground, -1 unknown)
            gt = cv2.imread(gt_dir, 0)
            gt_vec = gt.ravel()
            gt_vec[gt_vec <= 50] = 0
            gt_vec[(gt_vec >= 85) & (gt_vec <= 170)] = -1
            gt_vec[gt_vec > 170] = 1

            if idx == 0:
                frames = np.vstack((np.zeros([1, frame.shape[0] * frame.shape[1], 3]), frame_vec))
                gts = np.vstack((np.zeros([1, frame.shape[0] * frame.shape[1]]), gt_vec))
            else:
                frames = np.vstack((frames, frame_vec))
                gts = np.vstack((gts, gt_vec))

        frames = frames[1:, :, :]
        train_frames = frames[:int(round(frames.shape[0] * 0.5)), :, :]
        test_frames = frames[int(round(frames.shape[0] * 0.5)):, :, :]

    gts = gts[1:, :]
    train_gts = gts[:int(round(frames.shape[0] * 0.5)), :]
    test_gts = gts[int(round(frames.shape[0] * 0.5)):, :]

    return train_frames, test_frames, train_gts, test_gts


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

