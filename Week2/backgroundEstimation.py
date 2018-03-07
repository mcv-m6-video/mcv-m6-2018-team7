from __future__ import division
import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from AdaptiveClassifier import AdaptiveClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

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


def optimizeAlphaRho(alpha_values, rho_values, dataset_path, ID, Color):

    train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID, Color)
    frames = np.vstack((train_frames, test_frames))
    labels = np.vstack((train_gts, test_gts))
    train_idx = [range(int(round(frames.shape[0] * 0.5)))]
    test_idx = [range(int(round(frames.shape[0] * 0.5)), frames.shape[0])]

    init_mu = np.zeros([1, train_frames.shape[1]])
    init_std = np.zeros([1, train_frames.shape[1]])
    parameters = {'alpha': alpha_values, 'rho': rho_values}

    # perform grid search to optimize alpha and rho
    grid = GridSearchCV(AdaptiveClassifier(init_mu, init_std), parameters, cv=zip(train_idx, test_idx))
    grid.fit(frames, labels)

    # save results to disk
    f = open('gridsearch_'+ID+'.pckl', 'wb')
    pickle.dump(grid, f)
    f.close()

    return grid


def gridSearchAdaptiveClassifier(dataset_path, ID, Color):

    # define range of values for alpha and rho
    alpha_values = np.arange(0, 10, 0.2)
    rho_values = np.arange(0, 1, 0.1)

    # if gridsearch has already been computed just load results, otherwise compute it
    if os.path.isfile('gridsearch_'+ID+'.pckl'):
        f = open('grid.pckl', 'rb')
        grid = pickle.load(f)
        f.close()
    else:
        grid = optimizeAlphaRho(alpha_values, rho_values, dataset_path, ID, Color)

    # print optimal alpha and rho with the associated fscore
    print('Best parameters for ' + ID + ' dataset: %s Accuracy: %0.5f' % (grid.best_params_, grid.best_score_))

    # uncomment to see the score for each parameter combination
    #for fscore, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
    #    print("%0.3f for %r" % (fscore, params))

    # plot fscore surface (maximum will indicate best combination of parameters)
    fscores = grid.cv_results_['mean_test_score']
    X, Y = np.meshgrid(rho_values, alpha_values)
    Z = np.reshape(fscores, (len(alpha_values), len(rho_values)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("rho")
    ax.set_ylabel("alpha")
    ax.set_zlabel("F-score")
    ax.set_zlim(min(fscores), max(fscores))

    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                           rstride=1, cstride=1, linewidth=0, antialiased=True)
    fig.colorbar(surf)
    plt.show()

    return grid.best_params_['alpha'], grid.best_params_['rho'], grid.best_score_