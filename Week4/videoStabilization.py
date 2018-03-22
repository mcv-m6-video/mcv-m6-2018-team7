from __future__ import division
import numpy as np
import os
import cv2
from opticalFlowEstimation import calcOpticalFlowBM
from readInputWriteOutput import showOpticalFlowHSVBlockMatching, plotMotionEstimation
from smooth import smooth
import pickle


def videoStabilizationReference(data_path, blockSize, areaOfSearch, Backward, ID):

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

        #OF_image = calcOpticalFlowBM(curr_img, toExplore_img, blockSize, areaOfSearch)
        OF_image = cv2.calcOpticalFlowFarneback(curr_img, toExplore_img, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # showOpticalFlowHSVBlockMatching(OF_image, frames_list[idx], saveResult=True, visualization = 'HS')
        motionVectors[idx,:,:,0:2] = OF_image

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
        cv2.imwrite('results/Stabilized/'+ID+'/input/'+frames_list[idx], compensated)

    np.save('results/Stabilized/'+ID+'/motionVectors', motionVectors)
    np.save('results/Stabilized/'+ID+'/point', point)
    return motionVectors, point


def videoStabilizationPairs(data_path, blockSize, areaOfSearch, ID):

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

        cv2.imwrite('results/Stabilized/'+ID+'/input/'+frames_list[idx], compensated)

    return motionVectors


def videoStabilizationMovement(data_path, blockSize, areaOfSearch, Backward, ID):

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
        OF_image = cv2.calcOpticalFlowFarneback(curr_img, toExplore_img, pyr_scale=0.5, levels=3, winsize=30, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
        # showOpticalFlowHSVBlockMatching(OF_image, frames_list[idx], saveResult=True, visualization = 'HS')
        motionVectors[idx,:,:,0:2] = OF_image


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
        cv2.imwrite('results/Stabilized/'+ID+'/input/'+frames_list[idx], compensated)

    np.save('results/Stabilized/'+ID+'/motionVectors', motionVectors)
    np.save('results/Stabilized/'+ID+'/point', point)
    return motionVectors, point


def HomographyTransformStab(data_path):
    print '------Homography-based Video Stabilization--------'

    # List all the files in the sequence dataset
    frames_list = sorted(os.listdir(data_path))
    nFrames = len(frames_list)
    #compensated = cv2.imread(os.path.join(data_path, frames_list[0]), 0)
    ref = cv2.imread(os.path.join(data_path, frames_list[0]), 0)

    sift = cv2.SIFT()
    kp2, des2 = sift.detectAndCompute(ref, None)
    homo_matrix = np.ones([nFrames,2,3])
    for idx in range(0, nFrames):


        frame_dir = os.path.join(data_path, frames_list[idx])

        toCompensate = cv2.imread(frame_dir, 0)

        #M = cv2.estimateRigidTransform(toCompensate, ref, 0)
        kp1, des1 = sift.detectAndCompute(toCompensate, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2,k=2)
        nonOutliers = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                nonOutliers.append(m)
            sourcePoints = np.float32([kp1[mm.queryIdx].pt for mm in nonOutliers]).reshape(-1, 1, 2)
            destinationPoints = np.float32([kp2[mm.trainIdx].pt for mm in nonOutliers]).reshape(-1, 1, 2)
        M = cv2.estimateRigidTransform(sourcePoints, destinationPoints, 0)

        if M is not None:
            print '     --> Compensating frame ', frames_list[idx], '...'
            #s = (M[0][0] * M[0][0]) - (M[0][1] * M[0][1]) #calculate scaling factor
            #M = M/s #normalize
            M[0][0] = 1
            M[0][1] = 0
            M[1][0] = 0
            M[1][1] = 1
            compensated = cv2.warpAffine(toCompensate, M, (toCompensate.shape[1], toCompensate.shape[0])) #warp next into previous
            homo_matrix[:][:][idx] = M
        else:
            print 'Null transformation found'
            compensated = toCompensate
            homo_matrix[:][:][idx] = np.matrix([[1,0,0],[0,1,0]])

        cv2.imwrite('results/rectified_homo/'+frames_list[idx], compensated)
    f = open('homographies.pckl', 'wb')
    pickle.dump(homo_matrix, f)
    f.close()
