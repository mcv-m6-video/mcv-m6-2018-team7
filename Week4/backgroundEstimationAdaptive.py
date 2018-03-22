import numpy as np
import cv2
import os
from AdaptiveClassifier import AdaptiveClassifier
from sklearn.metrics import auc

def readDatasetStabilized(dataset_path, ID, Color):
    # Get the ID dataset path
    frames_path = 'results/Stabilized/'+ID+'/input'
    gt_path = dataset_path + ID + '/groundtruth'

    # List all the files in the ID dataset path
    frame_list = sorted(os.listdir(frames_path))
    gt_list = sorted(os.listdir(gt_path))
    print os.getcwd()
    motionVectors = np.load('results/Stabilized/'+ID+'/motionVectors.npy')
    point = np.load('results/Stabilized/'+ID+'/point.npy')

    if Color == False:
        # Compute mean and sigma  of each pixel
        for idx in range(len(frame_list)):

            # print 'Evaluating frame ' + pr_name
            frame_dir = os.path.join(frames_path, frame_list[idx])
            gt_dir = os.path.join(gt_path, gt_list[idx])

            # Get frame
            frame = cv2.imread(frame_dir, 0)

            M = np.float32([[1, 0, -motionVectors[idx, point[0], point[1], 0]],
                            [0, 1, -motionVectors[idx, point[0], point[1], 1]]])
            frame = cv2.warpAffine(frame.astype(float), M, (frame.shape[1], frame.shape[0]), borderValue=-1)

            frame_size = frame.shape

            frame_vec = frame.ravel()

            gt = cv2.imread(gt_dir, 0)
            gt = cv2.warpAffine(gt.astype(float), M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST, borderValue=85)
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
            frame_size = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    return train_frames, test_frames, train_gts, test_gts, frame_size

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

            frame_size = frame.shape

            frame_vec = frame.ravel()

            # Get groundtruth with three different values (0 background, 1 foreground, -1 unknown)
            gt = cv2.imread(gt_dir, 0)
            gt_vec = gt.ravel().astype(float)
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
            frame_size = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    return train_frames, test_frames, train_gts, test_gts, frame_size


def optimalAlphaAdaptive(dataset_path, ID, optimal_rho, holeFilling, areaFiltering, P, connectivity, Morph, shadRemov):

    print 'Computing optimal alpha for ' + ID + ' dataset ...'

    Precision_vec = []
    Recall_vec = []
    fscore_vec = []
    alpha_vec = np.arange(0, 30, 0.5)

    print 'Reading dataset...'
    train_frames_bw, test_frames_bw, train_gts, test_gts, frame_size = readDataset(dataset_path, ID, False)
    if shadRemov:
        train_frames_c, test_frames_c, train_gts, test_gts, frame_size = readDataset(dataset_path, ID, True)
    print 'Dataset has been read!'

    init_mu = np.zeros([1, train_frames_bw.shape[1]])
    init_std = np.zeros([1, train_frames_bw.shape[1]])
    init_mu_c = np.zeros([1, train_frames_bw.shape[1], 3])
    init_std_c = np.zeros([1, train_frames_bw.shape[1], 3])

    for alpha in alpha_vec:

        print 'Evaluating with alpha = ',alpha
        clf = AdaptiveClassifier(alpha, optimal_rho, init_mu, init_std, init_mu_c, init_std_c)
        clf = clf.fit(train_frames_bw,train_gts)

        if shadRemov:
            precision, recall, fscore = clf.performance_measures_shadowRemoval(train_frames_c, test_frames_bw,
                                                                               test_frames_c, test_gts, frame_size,
                                                                               holeFilling, areaFiltering, P,
                                                                               connectivity, Morph)
        else:
            precision, recall, fscore = clf.performance_measures(test_frames_bw, test_gts, frame_size,  holeFilling,
                                                                 areaFiltering, P, connectivity, Morph)

        Precision_vec = np.append(Precision_vec, precision)
        Recall_vec = np.append(Recall_vec, recall)
        fscore_vec = np.append(fscore_vec, fscore)

    min, max, idxmin, idxmax = cv2.minMaxLoc(fscore_vec)
    print 'Maximum F1-Score with ', ID, ' dataset is ', max, ' with alpha = ', alpha_vec[idxmax[1]]
    print 'Precision selected with dataset ', ID, ' is ', Precision_vec[idxmax[1]]
    print 'Recall selected with dataset ', ID, ' is ', Recall_vec[idxmax[1]]

    return Precision_vec, Recall_vec, fscore_vec, alpha_vec


def optimalAlphaAdaptiveStabilized(dataset_path, ID, optimal_rho, holeFilling, areaFiltering, P, connectivity, Morph, shadRemov):

    print 'Computing optimal alpha for ' + ID + ' dataset ...'

    Precision_vec = []
    Recall_vec = []
    fscore_vec = []
    alpha_vec = np.arange(0, 30, 0.5)

    print 'Reading dataset...'
    train_frames_bw, test_frames_bw, train_gts, test_gts, frame_size = readDatasetStabilized(dataset_path, ID, False)
    if shadRemov:
        train_frames_c, test_frames_c, train_gts, test_gts, frame_size = readDataset(dataset_path, ID, True)
    print 'Dataset has been read!'

    init_mu = np.zeros([1, train_frames_bw.shape[1]])
    init_std = np.zeros([1, train_frames_bw.shape[1]])
    init_mu_c = np.zeros([1, train_frames_bw.shape[1], 3])
    init_std_c = np.zeros([1, train_frames_bw.shape[1], 3])

    for alpha in alpha_vec:

        print 'Evaluating with alpha = ',alpha
        clf = AdaptiveClassifier(alpha, optimal_rho, init_mu, init_std, init_mu_c, init_std_c)
        clf = clf.fit(train_frames_bw,train_gts)

        if shadRemov:
            precision, recall, fscore = clf.performance_measures_shadowRemoval(train_frames_c, test_frames_bw,
                                                                               test_frames_c, test_gts, frame_size,
                                                                               holeFilling, areaFiltering, P,
                                                                               connectivity, Morph)
        else:
            precision, recall, fscore = clf.performance_measures(test_frames_bw, test_gts, frame_size,  holeFilling,
                                                                 areaFiltering, P, connectivity, Morph)

        Precision_vec = np.append(Precision_vec, precision)
        Recall_vec = np.append(Recall_vec, recall)
        fscore_vec = np.append(fscore_vec, fscore)

    min, max, idxmin, idxmax = cv2.minMaxLoc(fscore_vec)
    print 'Maximum F1-Score with ', ID, ' dataset is ', max, ' with alpha = ', alpha_vec[idxmax[1]]
    print 'Precision selected with dataset ', ID, ' is ', Precision_vec[idxmax[1]]
    print 'Recall selected with dataset ', ID, ' is ', Recall_vec[idxmax[1]]

    return Precision_vec, Recall_vec, fscore_vec, alpha_vec


def testAdaptiveClassifier(dataset_path, ID, alpha, rho, holeFilling, areaFiltering, P, connectivity, Morph, shadowRemoval):

    print 'Reading dataset...'
    train_frames_bw, test_frames_bw, train_gts, test_gts, frame_size = readDatasetStabilized(dataset_path, ID, False)
    if shadowRemoval:
        train_frames_c, test_frames_c, train_gts, test_gts, frame_size_c = readDataset(dataset_path, ID, True)
    print 'Dataset has been read!'

    init_mu = np.zeros([1, train_frames_bw.shape[1]])
    init_std = np.zeros([1, train_frames_bw.shape[1]])
    init_mu_c = np.zeros([1, train_frames_bw.shape[1], 3])
    init_std_c = np.zeros([1, train_frames_bw.shape[1], 3])

    clf = AdaptiveClassifier(alpha, rho, init_mu, init_std, init_mu_c, init_std_c)
    clf = clf.fit(train_frames_bw, train_gts)
    if shadowRemoval:
        precision, recall, fscore = clf.performance_measures_shadowRemoval(train_frames_c, test_frames_bw,
                                                                           test_frames_c, test_gts, frame_size,
                                                                           holeFilling, areaFiltering, P, connectivity, Morph)
    else:
        precision, recall, fscore = clf.performance_measures(test_frames_bw, test_gts, frame_size, holeFilling,
                                                             areaFiltering, P, connectivity, Morph)
    print 'F-score:', fscore
    print 'Precision: ', precision
    print 'Recall: ', recall
