from __future__ import division
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import skvideo.io


def readDataset(dataset_path, ID, Color, GT_available):

    print 'Reading dataset...'

    # Get frames path and list all frames
    frames_path = dataset_path + ID + '/input'
    frame_list = sorted(os.listdir(frames_path))
    # Load ground-truth if available
    if GT_available:
        gt_path = dataset_path + ID + '/groundtruth'
        gt_list = sorted(os.listdir(gt_path))

    # Read first frame
    frame_dir = os.path.join(frames_path, frame_list[0])
    frame_rgb = cv2.imread(frame_dir, 1)
    frame_gray = cv2.imread(frame_dir, 0)
    h, w, d = frame_rgb.shape
    frames_rgb = np.vstack((np.zeros([1, h * w, 3]), np.reshape(frame_rgb, (1, h * w, 3))))
    frames_gray = np.vstack((np.zeros([1, h * w]), frame_gray.ravel()))

    if GT_available:
        gt_dir = os.path.join(gt_path, gt_list[0])
        gt = cv2.imread(gt_dir, 0)
        gts = np.vstack((np.zeros([1, h * w]), gt.ravel()))

    for idx in range(1,len(frame_list)):

        # Read current frame
        #print 'Reading frame', idx
        frame_dir = os.path.join(frames_path, frame_list[idx])
        frame_rgb = cv2.imread(frame_dir, 1)
        frame_gray = cv2.imread(frame_dir, 0)
        h, w, d = frame_rgb.shape

        frames_rgb = np.vstack((frames_rgb, np.reshape(frame_rgb, (1, h * w, 3))))
        frames_gray = np.vstack((frames_gray, frame_gray.ravel()))

        if GT_available:
            gt_dir = os.path.join(gt_path, gt_list[idx])
            gt = cv2.imread(gt_dir, 0)
            gts = np.vstack((gts, gt.ravel()))

    if not GT_available:
        gts = np.zeros(frames_gray.shape)

    frames_rgb = frames_rgb[1:, :, :]
    train_rgb = frames_rgb[:int(round(frames_rgb.shape[0] * 0.5)), :, :]
    test_rgb = frames_rgb[int(round(frames_rgb.shape[0] * 0.5)):, :, :]

    frames_gray = frames_gray[1:, :]
    train_gray = frames_gray[:int(round(frames_gray.shape[0] * 0.5)), :]
    test_gray = frames_gray[int(round(frames_gray.shape[0] * 0.5)):, :]

    gts = gts[1:, :]
    # get ground-truth with three different values (0 background, 1 foreground, -1 unknown)
    gts[gts <= 50] = 0
    gts[(gts >= 85) & (gts <= 170)] = -1
    gts[gts > 170] = 1
    train_gts = gts[:int(round(frames_gray.shape[0] * 0.5)), :]
    test_gts = gts[int(round(frames_gray.shape[0] * 0.5)):, :]

    frame_size = (h, w)

    print 'Finished reading input frames!'

    if not Color:
        return train_gray, test_gray, train_gts, test_gts, frame_size
    else:
        return train_rgb, test_rgb, train_gray, test_gray, train_gts, test_gts, frame_size


def readDatasetStabilized(dataset_path, ID, Color, GT_available):

    print 'Reading dataset...'

    # Get frames path and list all frames
    frames_path = dataset_path + ID + '/input'
    frame_list = sorted(os.listdir(frames_path))

    motionVectors = np.load('results/Stabilized/' + ID + '/motionVectors.npy')
    point = np.load('results/Stabilized/' + ID + '/point.npy')

    # Load ground-truth if available
    if GT_available:
        gt_path = dataset_path + ID + '/groundtruth'
        gt_list = sorted(os.listdir(gt_path))

    # Read first frame

    M = np.float32([[1, 0, -motionVectors[0, point[0], point[1], 0]],
                    [0, 1, -motionVectors[0, point[0], point[1], 1]]])

    frame_dir = os.path.join(frames_path, frame_list[0])
    frame_rgb = cv2.imread(frame_dir, 1)
    frame_rgb = cv2.warpAffine(frame_rgb.astype(float), M, (frame_rgb.shape[1], frame_rgb.shape[0]), borderValue=-1)
    frame_gray = cv2.imread(frame_dir, 0)
    frame_gray = cv2.warpAffine(frame_gray.astype(float), M, (frame_gray.shape[1], frame_gray.shape[0]), borderValue=-1)
    h, w, d = frame_rgb.shape
    frames_rgb = np.vstack((np.zeros([1, h * w, 3]), np.reshape(frame_rgb, (1, h * w, 3))))
    frames_gray = np.vstack((np.zeros([1, h * w]), frame_gray.ravel()))

    if GT_available:
        gt_dir = os.path.join(gt_path, gt_list[0])
        gt = cv2.imread(gt_dir, 0)
        gt = cv2.warpAffine(gt.astype(float), M, (gt.shape[1], gt.shape[0]), borderValue=-1)
        gts = np.vstack((np.zeros([1, h * w]), gt.ravel()))

    for idx in range(1,len(frame_list)):

        M = np.float32([[1, 0, -motionVectors[idx, point[0], point[1], 0]],
                        [0, 1, -motionVectors[idx, point[0], point[1], 1]]])

        # Read current frame
        #print 'Reading frame', idx
        frame_dir = os.path.join(frames_path, frame_list[idx])
        frame_rgb = cv2.imread(frame_dir, 1)
        frame_gray = cv2.imread(frame_dir, 0)
        frame_rgb = cv2.warpAffine(frame_rgb.astype(float), M, (frame_rgb.shape[1], frame_rgb.shape[0]), borderValue=-1)
        frame_gray = cv2.warpAffine(frame_gray.astype(float), M, (frame_gray.shape[1], frame_gray.shape[0]), borderValue=-1)
        h, w, d = frame_rgb.shape

        frames_rgb = np.vstack((frames_rgb, np.reshape(frame_rgb, (1, h * w, 3))))
        frames_gray = np.vstack((frames_gray, frame_gray.ravel()))

        if GT_available:
            gt_dir = os.path.join(gt_path, gt_list[idx])
            gt = cv2.imread(gt_dir, 0)
            gt = cv2.warpAffine(gt.astype(float), M, (gt.shape[1], gt.shape[0]), borderValue=-1)
            gts = np.vstack((gts, gt.ravel()))

    if not GT_available:
        gts = np.zeros(frames_gray.shape)

    frames_rgb = frames_rgb[1:, :, :]
    train_rgb = frames_rgb[:int(round(frames_rgb.shape[0] * 0.5)), :, :]
    test_rgb = frames_rgb[int(round(frames_rgb.shape[0] * 0.5)):, :, :]

    frames_gray = frames_gray[1:, :]
    train_gray = frames_gray[:int(round(frames_gray.shape[0] * 0.5)), :]
    test_gray = frames_gray[int(round(frames_gray.shape[0] * 0.5)):, :]

    gts = gts[1:, :]
    # get ground-truth with three different values (0 background, 1 foreground, -1 unknown)
    gts[gts <= 50] = 0
    gts[(gts >= 85) & (gts <= 170)] = -1
    gts[gts > 170] = 1
    train_gts = gts[:int(round(frames_gray.shape[0] * 0.5)), :]
    test_gts = gts[int(round(frames_gray.shape[0] * 0.5)):, :]

    frame_size = (h, w)

    print 'Finished reading input frames!'

    if not Color:
        return train_gray, test_gray, train_gts, test_gts, frame_size
    else:
        return train_rgb, test_rgb, train_gray, test_gray, train_gts, test_gts, frame_size


def convertVideotoFrames(source_path, dest_path):

    vidcap = cv2.VideoCapture(source_path)
    success,image = vidcap.read()
    count = 0
    success = True

    while success:
      image = cv2.resize(image, (0,0), fx=0.8, fy=0.8)
      cv2.imwrite(dest_path+'frame_'+str(count).zfill(4)+'.jpg', image)     # save frame as JPEG file
      success,image = vidcap.read()
      print 'Read a new frame: ', success
      count += 1

    return


def downSampleSequence(source_path, factor_x, factor_y ):

    frames_path = source_path + '/input'
    dest_path = source_path + '/resized/'

    # List all the files in the ID dataset path
    frame_list = sorted(os.listdir(frames_path))

    for idx in range(len(frame_list)):
        print 'Reading frame ',str(idx)
        # print 'Evaluating frame ' + pr_name
        frame_dir = os.path.join(frames_path, frame_list[idx])
        frame = cv2.imread(frame_dir, 1)

        frame_res = cv2.resize(frame, (0,0), fx=factor_x, fy=factor_y)
        cv2.imwrite(dest_path+frame_list[idx], frame_res)     # save frame as JPEG file

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


def writeVideoFromFrames(frames_path, output_filename):

    frame_list = sorted(os.listdir(frames_path))

    frame = cv2.imread(os.path.join(frames_path, frame_list[0]), -1)
    nrows, ncols, nchannels = frame.shape
    n_frames = len(frame_list)

    data = np.zeros([n_frames, nrows, ncols, nchannels])  # [num_frames, rows, cols, channels]

    for idx in range(len(frame_list)):
        # print 'Evaluating frame ' + pr_name
        frame_dir = os.path.join(frames_path, frame_list[idx])
        frame = cv2.imread(frame_dir, 1)
        data[idx, :, :, :] = frame[:, :, :]

    skvideo.io.vwrite(output_filename, data)

