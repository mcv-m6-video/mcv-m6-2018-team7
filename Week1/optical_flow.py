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

    for pr_name, gt_name in zip(pred_list, gt_list):

        # print 'Evaluating frame '+ pr_name
        pred_file = os.path.join(pred_path, pr_name)
        gt_file = os.path.join(gt_path, gt_name)

        pr_OF = cv2.imread(pred_file, -1)
        gt_OF = cv2.imread(gt_file, -1)

        MSEN, PEPN = compute_MSEN_PEPN_errorImage(pr_OF.astype(float), gt_OF.astype(float), gt_name)


def compute_MSEN_PEPN_errorImage(pred_OF,gt_OF,gt_name):

    nrows, ncols, nchannels = pred_OF.shape
    errorImage = np.resize( np.zeros((nrows,ncols)), (nrows,ncols,nchannels))
    valid_gt_reshaped = np.resize(np.zeros((nrows, ncols)), (nrows, ncols, nchannels))

    #Read optical flow png images
    u_pred = (pred_OF[:,:,2].ravel() - math.pow(2, 15))  / 64.0
    v_pred = (pred_OF[:,:,1].ravel() - math.pow(2, 15))  / 64.0
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


def plotOpticalFlowHSV(OF_path):

    # Iterate through every image in the directory in alphabethic order
    OF_list = sorted(os.listdir(OF_path))

    for OF_name in OF_list:

        # print 'Evaluating frame '+ pr_name
        OF_file = os.path.join(OF_path, OF_name)
        OF_image = cv2.imread(OF_file, -1)

        nrows, ncols, nchannels = OF_image.shape
        valid_gt_reshaped = np.resize(np.zeros((nrows, ncols)), (nrows, ncols, nchannels))

        hsv = np.zeros([nrows, ncols, nchannels])
        hsv[:,:, 1] = 255

        u_vec = (OF_image[:, :, 2].ravel() - math.pow(2, 15)) / 64.0
        v_vec = (OF_image[:, :, 1].ravel() - math.pow(2, 15)) / 64.0
        valid_gt = OF_image[:, :, 0].ravel()

        u_vec = np.multiply(u_vec, valid_gt)
        v_vec = np.multiply(v_vec, valid_gt)

        u = np.reshape(u_vec, (nrows, ncols))
        v = np.reshape(v_vec, (nrows, ncols))

        mag, ang = cv2.cartToPolar(u, v)
        hsv[:,:, 0] = ang * 180 / np.pi /2
        hsv[:,:, 2] = cv2.normalize(mag, None, 50, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        valid_gt_reshaped[:, :, 2] = np.reshape(valid_gt, (nrows, ncols))
        valid_gt_reshaped[:, :, 1] = np.reshape(valid_gt, (nrows, ncols))
        valid_gt_reshaped[:, :, 0] = np.reshape(valid_gt, (nrows, ncols))
        rgb = np.multiply(rgb, valid_gt_reshaped)
        cv2.imwrite('OF_'+str(OF_name), rgb)