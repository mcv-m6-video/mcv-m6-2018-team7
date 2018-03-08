import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from AdaptiveClassifier import AdaptiveClassifier
from readInputWriteOutput import readDataset, precisionRecallCurve

def optimalAlphaAdaptive(dataset_path, ID, Color, optimal_rho):

    print 'Computing optimal alpha for ' + ID + ' dataset ...'

    Precision_vec = []
    Recall_vec = []
    fscore_vec = []
    alpha_vec = np.linspace(0, 30, 21)

    train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID, False)
    init_mu = np.zeros([1, train_frames.shape[1]])
    init_std = np.zeros([1, train_frames.shape[1]])

    for alpha in alpha_vec:

        clf = AdaptiveClassifier(alpha, optimal_rho, init_mu, init_std)
        clf = clf.fit(train_frames,train_gts)
        precision, recall, fscore = clf.performance_measures(test_frames, test_gts)

        Precision_vec = np.append(Precision_vec, precision)
        Recall_vec = np.append(Recall_vec, recall)
        fscore_vec = np.append(fscore_vec, fscore)

    min, max, idxmin, idxmax = cv2.minMaxLoc(fscore_vec)
    print 'Maximum F1-Score with ', ID, ' dataset is ', max, ' with alpha = ', alpha_vec[idxmax[1]]
    print 'Precision selected with dataset ', ID, ' is ', Precision_vec[idxmax[1]]
    print 'Recall selected with dataset ', ID, ' is ', Recall_vec[idxmax[1]]

    return Precision_vec, Recall_vec, fscore_vec, alpha_vec


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
    f = open('gridsearch_' + ID + '.pckl', 'wb')
    pickle.dump(grid, f)
    f.close()

    return grid


def gridSearchAdaptiveClassifier(dataset_path, ID, Color):

    # define range of values for alpha and rho
    alpha_values = np.linspace(0, 10, 21)
    rho_values = np.linspace(0, 1, 21)

    # if gridsearch has already been computed just load results, otherwise compute it
    if os.path.isfile('gridsearch_' + ID + '.pckl'):
        print 'Loading GridSearch...'
        f = open('gridsearch_' + ID + '.pckl', 'rb')
        grid = pickle.load(f)
        f.close()
    else:
        grid = optimizeAlphaRho(alpha_values, rho_values, dataset_path, ID, Color)

    # print optimal alpha and rho with the associated fscore
    print('Best parameters for ' + ID + ' dataset: %s Best F-score: %0.5f' % (grid.best_params_, grid.best_score_))

    # uncomment to see the score for each parameter combination
    # for fscore, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['params']):
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


def testAdaptiveClassifier(alpha, rho, dataset_path, ID):

    train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID, False)
    init_mu = np.zeros([1, train_frames.shape[1]])
    init_std = np.zeros([1, train_frames.shape[1]])
    clf = AdaptiveClassifier(alpha, rho, init_mu, init_std)
    clf = clf.fit(train_frames, train_gts)
    precision, recall, fscore = clf.performance_measures(test_frames, test_gts)
    print 'F-score:', fscore
    print 'Precision: ', precision
    print 'Recall: ', recall
