from __future__ import division
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def readDataset(dataset_path):

    # Get the ID dataset path
    frames_path = dataset_path + '/frames'
    gt_path = dataset_path + '/GT'

    # List all the files in the ID dataset path
    frame_list = sorted(os.listdir(frames_path))
    gt_list = sorted(os.listdir(gt_path))


    return frame_list, gt_list, frames_path, gt_path


def convertVideotoFrames(source_path, dest_path):

    vidcap = cv2.VideoCapture(source_path)
    success,image = vidcap.read()
    count = 0
    success = True

    while success:
      image = cv2.resize(image, (0,0), fx= 0.3, fy= 0.3)
      cv2.imwrite(dest_path+'frame_'+str(count).zfill(4)+'.jpg', image)     # save frame as JPEG file
      success,image = vidcap.read()
      print 'Read a new frame: ', success
      count += 1

    return


def showOpticalFlowHSVBlockMatching(OF_image, filename, saveResult, visualization):

    nrows, ncols, nchannels = OF_image.shape

    hsv = np.zeros([nrows, ncols, 3])

    if visualization == 'HS':
        hsv[:, :, 2] = 255
        mag, ang = cv2.cartToPolar(OF_image[:, :, 0], OF_image[:, :, 1])
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 1] = cv2.normalize(mag, None, 50, 255, cv2.NORM_MINMAX)

    if visualization == 'HV':
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(OF_image[:, :, 0], OF_image[:, :, 1])
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 2] = cv2.normalize(mag, None, 50, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    plt.figure()
    plt.imshow(rgb)
    plt.show()

    if saveResult:
        cv2.imwrite('results/OpticalFlow/images/OF_HSV_' + filename, rgb)
    return


def plotOpticalFlowHSV(OF_path, visualization = 'HS'):

    # Iterate through every image in the directory in alphabethic order
    OF_list = sorted(os.listdir(OF_path))

    for OF_name in OF_list:

        # print 'Evaluating frame '+ pr_name
        OF_file = os.path.join(OF_path, OF_name)
        OF_image = cv2.imread(OF_file, -1)

        nrows, ncols, nchannels = OF_image.shape
        valid_gt_reshaped = np.resize(np.zeros((nrows, ncols)), (nrows, ncols, nchannels))

        hsv = np.zeros([nrows, ncols, nchannels])

        if visualization == 'HS':
            hsv[:,:, 2] = 255

            u_vec = (OF_image[:, :, 2].ravel() - math.pow(2, 15)) / 64.0
            v_vec = (OF_image[:, :, 1].ravel() - math.pow(2, 15)) / 64.0
            valid_gt = OF_image[:, :, 0].ravel()

            u_vec = np.multiply(u_vec, valid_gt)
            v_vec = np.multiply(v_vec, valid_gt)

            u = np.reshape(u_vec, (nrows, ncols))
            v = np.reshape(v_vec, (nrows, ncols))

            mag, ang = cv2.cartToPolar(u, v)
            hsv[:,:, 0] = ang * 180 / np.pi /2
            hsv[:,:, 1] = cv2.normalize(mag, None, 50, 255, cv2.NORM_MINMAX)

        if visualization == 'HV':
            hsv[:, :, 1] = 255

            u_vec = (OF_image[:, :, 2].ravel() - math.pow(2, 15)) / 64.0
            v_vec = (OF_image[:, :, 1].ravel() - math.pow(2, 15)) / 64.0
            valid_gt = OF_image[:, :, 0].ravel()

            u_vec = np.multiply(u_vec, valid_gt)
            v_vec = np.multiply(v_vec, valid_gt)

            u = np.reshape(u_vec, (nrows, ncols))
            v = np.reshape(v_vec, (nrows, ncols))

            mag, ang = cv2.cartToPolar(u, v)
            hsv[:, :, 0] = ang * 180 / np.pi / 2
            hsv[:, :, 2] = cv2.normalize(mag, None, 50, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        valid_gt_reshaped[:, :, 2] = np.reshape(valid_gt, (nrows, ncols))
        valid_gt_reshaped[:, :, 1] = np.reshape(valid_gt, (nrows, ncols))
        valid_gt_reshaped[:, :, 0] = np.reshape(valid_gt, (nrows, ncols))
        rgb = np.multiply(rgb, valid_gt_reshaped)
        cv2.imwrite('GT_OF_'+str(OF_name), rgb)


def showOpticalFlowArrows(OF_image, img, filename, saveResult):


    X, Y = np.meshgrid(np.arange(0, OF_image.shape[1]), np.arange(0, OF_image.shape[0]))
    U = OF_image[:,:,0]
    V = OF_image[:,:,1]

    SF = 15 # Sampling Factor
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('Arrows scale with plot width, not view')
    Q = plt.quiver(X[::SF, ::SF], Y[::SF, ::SF], U[::SF, ::SF], V[::SF, ::SF], color = [1,0.75,0], width=1, units = 'xy')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')

    plt.show()

    if saveResult:
        fig.savefig('results/OpticalFlow/images/OF_arrows_' + filename)
        plt.close()
    return



def plotMSENPEPNCurves(MSEN_img1, MSEN_img2, PEPN_img1, PEPN_img2, blockSize_vec, areaOfSearch_vec):


    # Plot PEPN sequence 45
    fig = plt.figure()
    fig.suptitle('PEPN vs Area of search', fontsize=15)
    ax3 = fig.add_subplot(111)
    ax3.set_xlabel('Area of search', fontsize=11)
    ax3.set_ylabel('PEPN', fontsize=11)
    ax3.set_xlim([0,np.max(areaOfSearch_vec)])
    ax3.plot(areaOfSearch_vec, PEPN_img1[0,:], c='orange', linewidth=3.0, label ='BlockSize = 20')
    ax3.plot(areaOfSearch_vec, PEPN_img1[1,:], c='green', linewidth=3.0, label='BlockSize = 30')
    ax3.plot(areaOfSearch_vec, PEPN_img1[2, :], c='red', linewidth=3.0, label='BlockSize = 40')
    ax3.plot(areaOfSearch_vec, PEPN_img1[3, :], c='blue', linewidth=3.0, label='BlockSize = 50')
    ax3.plot(areaOfSearch_vec, PEPN_img1[4, :], c='brown', linewidth=3.0, label='BlockSize = 60')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotPEPNCurves_seq000045.png', bbox_inches='tight')

    # Plot PEPN sequence 157
    fig = plt.figure()
    fig.suptitle('PEPN vs Area of search', fontsize=15)
    ax3 = fig.add_subplot(111)
    ax3.set_xlabel('Area of search', fontsize=11)
    ax3.set_ylabel('PEPN', fontsize=11)
    ax3.set_xlim([0, np.max(areaOfSearch_vec)])
    ax3.plot(areaOfSearch_vec, PEPN_img2[0,:], c='orange', linewidth=3.0, label ='BlockSize = 20')
    ax3.plot(areaOfSearch_vec, PEPN_img2[1,:], c='green', linewidth=3.0, label='BlockSize = 30')
    ax3.plot(areaOfSearch_vec, PEPN_img2[2, :], c='red', linewidth=3.0, label='BlockSize = 40')
    ax3.plot(areaOfSearch_vec, PEPN_img2[3, :], c='blue', linewidth=3.0, label='BlockSize = 50')
    ax3.plot(areaOfSearch_vec, PEPN_img2[4, :], c='brown', linewidth=3.0, label='BlockSize = 60')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotPEPNCurves_seq0000157.png', bbox_inches='tight')

    # Plot MSEN sequence 45
    fig = plt.figure()
    fig.suptitle('MSEN vs Area of search', fontsize=15)
    ax3 = fig.add_subplot(111)
    ax3.set_xlabel('Area of search', fontsize=11)
    ax3.set_ylabel('MSEN', fontsize=11)
    ax3.set_xlim([0,np.max(areaOfSearch_vec)])
    ax3.plot(areaOfSearch_vec, MSEN_img1[0,:], c='orange', linewidth=3.0, label ='BlockSize = 20')
    ax3.plot(areaOfSearch_vec, MSEN_img1[1,:], c='green', linewidth=3.0, label='BlockSize = 30')
    ax3.plot(areaOfSearch_vec, MSEN_img1[2, :], c='red', linewidth=3.0, label='BlockSize = 40')
    ax3.plot(areaOfSearch_vec, MSEN_img1[3, :], c='blue', linewidth=3.0, label='BlockSize = 50')
    ax3.plot(areaOfSearch_vec, MSEN_img1[4, :], c='brown', linewidth=3.0, label='BlockSize = 60')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotMSENCurves_seq000045.png', bbox_inches='tight')

    # Plot MSEN sequence 157
    fig = plt.figure()
    fig.suptitle('MSEN vs Area of search', fontsize=15)
    ax3 = fig.add_subplot(111)
    ax3.set_xlabel('Area of search', fontsize=11)
    ax3.set_ylabel('MSEN', fontsize=11)
    ax3.set_xlim([0, np.max(areaOfSearch_vec)])
    ax3.plot(areaOfSearch_vec, MSEN_img2[0,:], c='orange', linewidth=3.0, label ='BlockSize = 20')
    ax3.plot(areaOfSearch_vec, MSEN_img2[1,:], c='green', linewidth=3.0, label='BlockSize = 30')
    ax3.plot(areaOfSearch_vec, MSEN_img2[2, :], c='red', linewidth=3.0, label='BlockSize = 40')
    ax3.plot(areaOfSearch_vec, MSEN_img2[3, :], c='blue', linewidth=3.0, label='BlockSize = 50')
    ax3.plot(areaOfSearch_vec, MSEN_img2[4, :], c='brown', linewidth=3.0, label='BlockSize = 60')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotMSENCurves_seq0000157.png', bbox_inches='tight')


    return


def plotMotionEstimation(motionVectors, point, name):

    nFrames = motionVectors.shape[0]
    idx = np.arange(0,nFrames)

    # Plot PEPN sequence 45
    fig = plt.figure()
    fig.suptitle('Motion estimation along frames', fontsize=15)
    ax3 = fig.add_subplot(111)
    ax3.set_xlabel('Frame', fontsize=11)
    ax3.set_ylabel('Pixels', fontsize=11)
    ax3.set_xlim([0, nFrames])
    ax3.set_ylim([-30, +30])
    ax3.plot(idx, motionVectors[idx,point[0],point[1],0], c='orange', linewidth=3.0, label='U component')
    ax3.plot(idx, motionVectors[idx,point[0],point[1],1], c='green', linewidth=3.0, label='V component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('plotMotionEstimation_'+name+'.png', bbox_inches='tight')

    return


def readFLO(file):

	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file,'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == 202021.25, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = np.fromfile(f, np.int32, count=1)
	h = np.fromfile(f, np.int32, count=1)
	#if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	data = np.fromfile(f, np.float32, count=2*w*h)
	# Reshape data into 3D array (columns, rows, bands)
	flow = np.resize(data, (int(h), int(w), 2))
	f.close()

	return flow