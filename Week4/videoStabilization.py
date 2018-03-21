from __future__ import division
import numpy as np
import os
import cv2
from opticalFlowEstimation import calcOpticalFlowBM
from readInputWriteOutput import showOpticalFlowHSVBlockMatching, plotMotionEstimation
from smooth import smooth


def videoStabilizationReference(data_path, blockSize, areaOfSearch, Backward):

    print '------Video Stabilization--------'

    # List all the files in the sequence dataset
    frames_list = sorted(os.listdir(data_path))
    nFrames = len(frames_list)

    frame_dir_0 = os.path.join(data_path, frames_list[0])
    first_frame = cv2.imread(frame_dir_0,0)

    motionVectors = np.zeros([nFrames, first_frame.shape[0], first_frame.shape[1], 3])

    print 'Motion estimation step'
    for idx in range(0, nFrames):

        print '     --> Analyzing frame ', frames_list[idx],'...'
        frame_dir = os.path.join(data_path, frames_list[idx])

        if Backward:
            toExplore_img = first_frame
            curr_img = cv2.imread(frame_dir, 0)

        else:
            curr_img = first_frame
            toExplore_img = cv2.imread(frame_dir, 0)

        # OF_image = calcOpticalFlowBM(curr_img, toExplore_img, blockSize, areaOfSearch)
        OF_image = cv2.calcOpticalFlowFarneback(curr_img, toExplore_img, pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # showOpticalFlowHSVBlockMatching(OF_image, frames_list[idx], saveResult=True, visualization = 'HS')
        motionVectors[idx] = OF_image

    point = [217,75] # Reference point from which the sequence is stabilized
    # plotMotionEstimation(motionVectors, point)

    print ' -->Motion estimation completed!'

    print 'Motion compensation step'
    for idx in range(0, nFrames):

        print '     --> Compensating frame ', frames_list[idx],'...'
        frame_dir = os.path.join(data_path, frames_list[idx])

        toCompensate = cv2.imread(frame_dir, 1)
        M = np.float32([[1, 0, -motionVectors[idx][150,105,0]], [0, 1, -motionVectors[idx][150,105,1]]])
        compensated = cv2.warpAffine(toCompensate, M, (first_frame.shape[1], first_frame.shape[0]))
        cv2.imwrite('results/Stabilized/'+frames_list[idx], compensated)

    np.save('results/Stabilized/motionVectors', motionVectors)
    np.save('results/Stabilized/point', point)
    return motionVectors, point


def videoStabilizationPairs(data_path, blockSize, areaOfSearch):

    print '------Video Stabilization--------'

    # List all the files in the sequence dataset
    frames_list = sorted(os.listdir(data_path))
    nFrames = len(frames_list)

    first_frame_dir = os.path.join(data_path, frames_list[0])
    first_frame = cv2.imread(first_frame_dir,0)
    compensated = first_frame

    motionVectors = np.zeros([nFrames, first_frame.shape[0], first_frame.shape[1], 3])

    print 'Motion estimation step'
    for idx in range(0, nFrames):

        print '     --> Analyzing frame ', frames_list[idx],'...'
        toCompensate_dir = os.path.join(data_path, frames_list[idx])

        curr_img = compensated
        toCompensate = cv2.imread(toCompensate_dir, 0)

        OF_image = calcOpticalFlowBM(curr_img, toCompensate, blockSize, areaOfSearch)
        # OF_image = cv2.calcOpticalFlowFarneback(curr_img, toCompensate, pyr_scale=0.5, levels=3, winsize=30,
        #                                          iterations=5, poly_n=5, poly_sigma=1.2, flags=0)

        # Motion compensation on the actual frame
        motionVectors[idx] = OF_image
        M = np.float32([[1, 0, -motionVectors[idx][138, 83, 0]], [0, 1, -motionVectors[idx][138, 83, 1]]])
        compensated = cv2.warpAffine(toCompensate, M, (first_frame.shape[1], first_frame.shape[0]))

        cv2.imwrite('results/Stabilized/'+frames_list[idx], compensated)

    return motionVectors


def videoStabilizationMovement(data_path, blockSize, areaOfSearch, Backward):

    print '------Video Stabilization--------'

    # List all the files in the sequence dataset
    frames_list = sorted(os.listdir(data_path))
    nFrames = len(frames_list)

    frame_dir_0 = os.path.join(data_path, frames_list[0])
    first_frame = cv2.imread(frame_dir_0,0)

    motionVectors = np.zeros([nFrames, first_frame.shape[0], first_frame.shape[1], 2])

    print 'Motion estimation step'
    for idx in range(0, nFrames-1):

        print '     --> Analyzing frame ', frames_list[idx],'...'
        frame_dir_curr = os.path.join(data_path, frames_list[idx])
        frame_dir_next = os.path.join(data_path, frames_list[idx+1])

        if Backward:
            toExplore_img = cv2.imread(frame_dir_curr, 0)
            curr_img = cv2.imread(frame_dir_next, 0)

        else:
            curr_img = cv2.imread(frame_dir_curr, 0)
            toExplore_img = cv2.imread(frame_dir_next, 0)

        #OF_image = calcOpticalFlowBM(curr_img, toExplore_img, blockSize, areaOfSearch)
        OF_image = cv2.calcOpticalFlowFarneback(curr_img, toExplore_img, pyr_scale=0.5, levels=3, winsize=30,
                                                iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
        # showOpticalFlowHSVBlockMatching(OF_image, frames_list[idx], saveResult=True, visualization = 'HS')
        motionVectors[idx] = OF_image


    point = [138,83] # Reference point from which the sequence is stabilized
    plotMotionEstimation(motionVectors, point, name= 'beforeSmoothing')
    motionVectors[:, 138, 83, 0] = smooth(motionVectors[:, 138, 83, 0], window_len= 1)
    motionVectors[:, 138, 83, 1] = smooth(motionVectors[:, 138, 83, 1], window_len=1)
    plotMotionEstimation(motionVectors, point, name= 'afterSmoothing')

    print ' -->Motion estimation completed!'

    print 'Motion compensation step'
    for idx in range(0, nFrames):

        print '     --> Compensating frame ', frames_list[idx],'...'
        frame_dir = os.path.join(data_path, frames_list[idx])

        toCompensate = cv2.imread(frame_dir, 1)
        M = np.float32([[1, 0, -motionVectors[idx][128,83,0]], [0, 1, -motionVectors[idx][138,83,1]]])
        compensated = cv2.warpAffine(toCompensate, M, (first_frame.shape[1], first_frame.shape[0]))
        cv2.imwrite('results/Stabilized/'+frames_list[idx], compensated)

    np.save('results/Stabilized/motionVectors', motionVectors)
    np.save('results/Stabilized/point', point)
    return motionVectors, point