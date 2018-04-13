from __future__ import division
import numpy as np
import cv2
import os
from opticalFlowEstimation import calcOpticalFlowBM

def videoStabilizationReference(data_path, blockSize, areaOfSearch, Backward, ID, point):

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

        OF_image = calcOpticalFlowBM(curr_img, toExplore_img, blockSize, areaOfSearch)
        # OF_image = cv2.calcOpticalFlowFarneback(curr_img, toExplore_img, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # showOpticalFlowHSVBlockMatching(OF_image, frames_list[idx], saveResult=True, visualization = 'HS')
        motionVectors[idx,:,:,0:3] = OF_image

    print ' -->Motion estimation completed!'

    print 'Motion compensation step'
    for idx in range(0, nFrames):

        print '     --> Compensating frame ', frames_list[idx],'...'
        frame_dir = os.path.join(data_path, frames_list[idx])

        toCompensate = cv2.imread(frame_dir, 1)
        M = np.float32([[1, 0, -motionVectors[idx, point[0],point[1],0]], [0, 1, -motionVectors[idx,point[0],point[1],1]]])
        compensated = cv2.warpAffine(toCompensate, M, (first_frame.shape[1], first_frame.shape[0]))
        cv2.imwrite('Results/'+ID+'/'+frames_list[idx], compensated)

    # np.save('results/'+ID+'/motionVectors', motionVectors)
    # np.save('results/Stabilized/'+ID+'/point', point)
    return motionVectors, point
