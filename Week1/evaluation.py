from __future__ import division
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


def readFolder(pred_path, gt_path):

    #Iterate through every image in the directory in alphabethic order
    pred_list = sorted(os.listdir(pred_path))
    gt_list = sorted(os.listdir(gt_path))

    pred_vec = []
    gt_vec = []

    for pr_name, gt_name in zip(pred_list, gt_list):

        #print 'Evaluating frame '+ pr_name
        pred_dir = os.path.join(pred_path,pr_name)
        gt_dir = os.path.join(gt_path,gt_name)

        pr_img = cv2.imread(pred_dir, 0)
        pr_img = pr_img.ravel()

        gt_img = cv2.imread(gt_dir, 0)
        gt_img = gt_img.ravel()
        gt_img = (gt_img >= 170).astype(int)

        pred_vec = np.concatenate([pred_vec, pr_img])
        gt_vec = np.concatenate([gt_vec, gt_img])

    return pred_vec, gt_vec


def evaluateFolder(pred_vec,gt_vec):

    precision, recall, fscore, support = precision_recall_fscore_support(gt_vec, pred_vec, average='binary')

    #TP = sum(((pred_vec > 0) & (gt_vec > 0)).astype(int))
    #FP = sum(((pred_vec > 0) & (gt_vec == 0)).astype(int))
    #FN = sum(((pred_vec == 0) & (gt_vec > 0)).astype(int))
    #TN = sum(((pred_vec == 0) & (gt_vec == 0)).astype(int))
    #precision = TP / (TP+FP)
    #recall = TP / (TP+FN)
    #fscore = 2*TP / (2*TP + FN + FP)

    print 'Precision: ', precision
    print 'Recall: ', recall
    print 'F1 score: ', fscore
    return precision, recall, fscore


def readFrame(frame_path,gt_path):

    frame = cv2.imread(frame_path, 0)
    gt = cv2.imread(gt_path, 0)
    gt = (gt == 255).astype(int)

    return frame, gt


def evaluateFrame(frame, gt):

    #confMat = confusion_matrix(gt.ravel(), frame.ravel())
    precision, recall, fscore, support = precision_recall_fscore_support(gt.ravel(), frame.ravel(),average='binary')

    TP = np.sum(np.logical_and(gt.ravel(), frame.ravel()).astype(int))
    TF = np.sum((gt == 1).astype(int))
    #TF = np.sum(frame.ravel()).astype(int)

    return TP, TF, precision, recall, fscore


def evaluateTemporal(pred_path, gt_path):

    #Iterate through every image in the directory in alphabethic order
    pred_list = sorted(os.listdir(pred_path))
    gt_list = sorted(os.listdir(gt_path))

    TP_vec = []
    TF_vec = []
    FS_vec = []
    for pr_name, gt_name in zip(pred_list, gt_list):

        #print 'Evaluating frame '+ pr_name
        pred_dir = os.path.join(pred_path,pr_name)
        gt_dir = os.path.join(gt_path,gt_name)
        pred , gt = readFrame(pred_dir, gt_dir)
        TP, TF, precision, recall, fscore = evaluateFrame(pred, gt)
        TP_vec = np.append(TP_vec, TP)
        TF_vec = np.append(TF_vec, TF)
        FS_vec = np.append(FS_vec, fscore)

    #Plot TP TF
    fig = plt.figure()
    fig.suptitle('Detected / Real Foreground per frame', fontsize=15)
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Pixels', fontsize=11)
    ax1.set_xlabel('Frame', fontsize=11)
    ax1.plot(range(1,np.size(TP_vec)+1), TP_vec, c='b', label= 'TP')
    ax2 = fig.add_subplot(111)
    ax2.plot(range(1, np.size(TF_vec)+1), TF_vec, c='r',label= 'TF')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotTPTF.png',bbox_inches='tight')
    plt.close()

    #Plot F1 per frame
    fig2 = plt.figure()
    fig2.suptitle('F1 score per frame', fontsize=15)
    ax3 = fig2.add_subplot(111)
    ax3.set_ylabel('F1 score', fontsize=11)
    ax3.set_xlabel('Frame', fontsize=11)
    ax3.plot(range(1,np.size(FS_vec)+1), FS_vec, c='b')
    fig2.savefig('plotFscore.png')
    plt.close()

def dSyncGlobalEvaluation(pred_vecA, pred_vecB, gt_vec):

    fscore_vecA = np.array([])
    fscore_vecB = np.array([])
    #de-synchronize from 0 to 25 frames
    for i in range(0,25):
        end = np.size(gt_vec)
        aux_pred_vecA = pred_vecA[i*(240*320):end]
        aux_pred_vecB = pred_vecB[i * (240 * 320):end]
        aux_gt_vec = gt_vec[0:end-i*(240*320)]
        precision, recall, fscoreA, support = precision_recall_fscore_support(aux_gt_vec, aux_pred_vecA, average='binary')
        precision, recall, fscoreB, support = precision_recall_fscore_support(aux_gt_vec, aux_pred_vecB, average='binary')
        fscore_vecA = np.append(fscore_vecA, fscoreA)
        fscore_vecB = np.append(fscore_vecB, fscoreB)

    fig = plt.figure()
    fig.suptitle('F1-score per number of de-synchronized frames', fontsize=15)
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('F1-sore', fontsize=11)
    ax1.set_xlabel('# of de-sync frames', fontsize=11)
    ax1.plot(range(1, np.size(fscore_vecA) + 1), fscore_vecA, c='b')
    ax1.plot(range(1, np.size(fscore_vecB) + 1), fscore_vecB, c='r')
    fig.savefig('plotDsync.png')
    plt.close()


# evaluation per frame for different syncronization delays and plot
def dSyncTemporalEvaluation(pred_path, gt_path):

    #Iterate through every image in the directory in alphabethic order
    pred_list = sorted(os.listdir(pred_path))
    gt_list = sorted(os.listdir(gt_path))

    fscore_list = []
    delays = (0, 5, 10, 20, 25)
    for d in delays:
        FS_vec = []
        for i in range(0, np.size(pred_list) - 25):
            pr_name = pred_list[i+d]
            gt_name = gt_list[i]
            #print 'Evaluating frame '+ pr_name
            pred_dir = os.path.join(pred_path,pr_name)
            gt_dir = os.path.join(gt_path,gt_name)
            pred , gt = readFrame(pred_dir, gt_dir) #llegir gt desplasat
            TP, TF, precision, recall, fscore = evaluateFrame(pred, gt)
            FS_vec = np.append(FS_vec, fscore)
        fscore_list.append(FS_vec)

    #Plot F1 per frame per each de-synchronization ammount
    fig = plt.figure()
    fig.suptitle('F1 score per frame with d-sync', fontsize=15)

    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('F1 score', fontsize=11)
    ax1.set_xlabel('Frame', fontsize=11)
    ax1.plot(range(1,np.size(fscore_list[0])+1), fscore_list[0], c='b', label='0 frames d-sync')

    ax2 = fig.add_subplot(111)
    ax2.set_ylabel('F1 score', fontsize=11)
    ax2.set_xlabel('Frame', fontsize=11)
    ax2.plot(range(1,np.size(fscore_list[1])+1), fscore_list[1], c='r', label='5 frames d-sync')

    ax3 = fig.add_subplot(111)
    ax3.set_ylabel('F1 score', fontsize=11)
    ax3.set_xlabel('Frame', fontsize=11)
    ax3.plot(range(1,np.size(fscore_list[2])+1), fscore_list[2], c='g', label='10 frames d-sync')

    ax4 = fig.add_subplot(111)
    ax4.set_ylabel('F1 score', fontsize=11)
    ax4.set_xlabel('Frame', fontsize=11)
    ax4.plot(range(1,np.size(fscore_list[3])+1), fscore_list[3], c='y', label='20 frames d-sync')

    ax5 = fig.add_subplot(111)
    ax5.set_ylabel('F1 score', fontsize=11)
    ax5.set_xlabel('Frame', fontsize=11)
    ax5.plot(range(1,np.size(fscore_list[4])+1), fscore_list[4], c='m', label='25 frames d-sync')

    # ax6 = fig.add_subplot(111)
    # ax6.set_ylabel('F1 score', fontsize=11)
    # ax6.set_xlabel('Frame', fontsize=11)
    # ax6.plot(range(1,np.size(fscore_list[5])+1), fscore_list[5], c='c', label='5 frames d-sync')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()
    fig.savefig('plotFscoreDsync.png',bbox_inches='tight')
    plt.close()
