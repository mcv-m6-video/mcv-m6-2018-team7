from __future__ import division
import os
import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def evaluateOpticalFlow(pred_path, gt_path):

    # Iterate through every image in the directory in alphabethic order
    pred_list = sorted(os.listdir(pred_path))
    gt_list = sorted(os.listdir(gt_path))

    MSEN =[]; PEPN = []
    for pr_name, gt_name in zip(pred_list, gt_list):

        # print 'Evaluating frame '+ pr_name
        pred_file = os.path.join(pred_path, pr_name)
        gt_file = os.path.join(gt_path, gt_name)

        pr_OF = np.load(pred_file)
        gt_OF = cv2.imread(gt_file, -1)

        msen, pepn = compute_MSEN_PEPN_errorImage(pr_OF.astype(float), gt_OF.astype(float), gt_name, saveHistogram = False)
        MSEN = np.append(MSEN, msen)
        PEPN = np.append(PEPN, pepn)

    return MSEN, PEPN


def compute_MSEN_PEPN_errorImage(pred_OF,gt_OF,gt_name, saveHistogram):

    nrows, ncols, nchannels = pred_OF.shape
    errorImage = np.resize( np.zeros((nrows,ncols)), (nrows,ncols,nchannels))
    valid_gt_reshaped = np.resize(np.zeros((nrows, ncols)), (nrows, ncols, nchannels))

    #Read optical flow png images
    u_pred = (pred_OF[:,:,0].ravel())
    v_pred = (pred_OF[:,:,1].ravel())
    u_gt = (gt_OF[:, :, 2].ravel() - math.pow(2, 15)) / 64.0
    v_gt = (gt_OF[:, :, 1].ravel() - math.pow(2, 15)) / 64.0
    valid_gt = gt_OF[:,:,0].ravel()

    #Compute error
    squared_error = np.sqrt( np.power((u_gt - u_pred), 2) + np.power((v_gt - v_pred), 2) )
    # Preserve only the values in the valid areas of the ground truth
    errorImage_vec = np.multiply(valid_gt, squared_error)

    #Compute incorrect predictions
    incorrect_pred = (errorImage_vec > 3.0).astype(int)

    #Compute PEPN and MSEN
    PEPN = sum(incorrect_pred)/sum(valid_gt) * 100
    MSEN = sum(errorImage_vec)/sum(valid_gt)
    print 'Mean Squared Error in Non-occluded areas: ', MSEN
    print 'Percentage of Erroneous Pixels in Non-occluded areas: ', PEPN

    if saveHistogram:
        #Visualize in which parts of the image the error is smaller/larger
        errorImage_vec = cv2.normalize(errorImage_vec, None, 50, 255, cv2.NORM_MINMAX)
        error_reshaped = np.reshape(errorImage_vec, (nrows,ncols))
        errorImage[:, :, 2] = error_reshaped
        errorImage[:, :, 1] = error_reshaped
        errorImage[:, :, 0] = error_reshaped
        errorImage = errorImage.astype(np.uint8)
        errorImage = cv2.applyColorMap(errorImage.astype(np.uint8), cv2.COLORMAP_JET)
        valid_gt_reshaped[:, :, 2] = np.reshape(valid_gt, (nrows, ncols))
        valid_gt_reshaped[:, :, 1] = np.reshape(valid_gt, (nrows, ncols))
        valid_gt_reshaped[:, :, 0] = np.reshape(valid_gt, (nrows, ncols))
        errorImage = np.multiply(valid_gt_reshaped, errorImage)
        cv2.imwrite('errorImage_'+gt_name, errorImage)

        #Save histogram of errors
        plt.hist(squared_error[valid_gt == 1], 40, normed=True)
        plt.axvline(x=MSEN, color='r', linestyle='--', linewidth=3.0)
        plt.title("Optical Flow Error Histogram")
        plt.xlabel("Mean Square Error")
        plt.ylabel("Pixel percentage")
        plt.savefig('errorHistogram_'+gt_name)
        plt.close()

    return MSEN, PEPN


