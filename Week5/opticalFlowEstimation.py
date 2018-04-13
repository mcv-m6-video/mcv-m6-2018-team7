from __future__ import division
import numpy as np
import os
import cv2
from readInputWriteOutput import readDataset, showOpticalFlowHSVBlockMatching, showOpticalFlowArrows


def euclidianDistance(current_block, search_block):

    dist = sum(sum(pow(current_block-search_block,2)))

    return dist


def blockMatching(curr_img, toExplore_img, blockSize, areaOfSearch):

    nRows, nCols = curr_img.shape

    u = np.zeros([nRows,nCols])
    v = np.zeros([nRows,nCols])

    # Iterate through each block in the current image
    for colIdx in range(0, nCols, blockSize):
        for rowIdx in range(0, nRows, blockSize):

            curr_block = curr_img[rowIdx:rowIdx+blockSize, colIdx:colIdx+blockSize]

            dist_vec = [];  v_vec = []; u_vec = []

            # For each block iterate through all possible matches on the searching image
            for colIdx_s in range(np.max((0, colIdx-areaOfSearch)), np.min((nCols-blockSize, colIdx+areaOfSearch))):
                for rowIdx_s in range(np.max((0, rowIdx-areaOfSearch)), np.min((nRows-blockSize, rowIdx+areaOfSearch))):

                    search_block = toExplore_img[rowIdx_s:rowIdx_s+blockSize, colIdx_s:colIdx_s+blockSize]

                    # Save euclidian distance between current block and each possible correspondecen and their corresponding OF vector
                    dist_vec =np.append(dist_vec, euclidianDistance(curr_block, search_block))
                    u_vec = np.append(u_vec, colIdx_s - colIdx)
                    v_vec = np.append(v_vec, rowIdx_s - rowIdx)

            # Get the OF vector corresponding to the more similar patch
            min, max, minLoc, maxLoc = cv2.minMaxLoc(dist_vec)
            u[rowIdx:rowIdx+blockSize, colIdx:colIdx+blockSize] = u_vec[minLoc[1]]
            v[rowIdx:rowIdx + blockSize, colIdx:colIdx + blockSize] = v_vec[minLoc[1]]

    return u, v


def calcOpticalFlowBM (curr_img, toExplore_img, blockSize, areaOfSearch):

    OF_image = np.zeros([curr_img.shape[0], curr_img.shape[1], 3])
    # Number of blocks in each dimension given img size and blockSize
    x_blocks = int(np.ceil(toExplore_img.shape[0] / blockSize))
    y_blocks = int(np.ceil(toExplore_img.shape[1] / blockSize))

    # Add padding in both imageS
    curr_img_pad = np.zeros([x_blocks * blockSize, y_blocks * blockSize])
    toExplore_img_pad = np.zeros([x_blocks * blockSize, y_blocks * blockSize])
    curr_img_pad[0:toExplore_img.shape[0], 0:toExplore_img.shape[1]] = curr_img[:, :]
    toExplore_img_pad[0:toExplore_img.shape[0], 0:toExplore_img.shape[1]] = toExplore_img[:, :]

    # Block Matching search
    u, v = blockMatching(curr_img_pad, toExplore_img_pad, blockSize, areaOfSearch)

    OF_image[:, :, 0] = u[:curr_img.shape[0], :curr_img.shape[1]]
    OF_image[:, :, 1] = v[:curr_img.shape[0], :curr_img.shape[1]]

    return OF_image


def estimateOpticalFlow(dataset_path, blockSize, areaOfSearch, Backward, showResult):

    print 'Optical Flow computation with BlockSize = ',blockSize, ' and AreaOfSearch = ',areaOfSearch
    frames_list, gt_list, frames_path, gt_path = readDataset(dataset_path)
    nFrames = len(frames_list)

    for idx in range(0, nFrames, 2):

        print '     --> Analyzing sequence ',frames_list[idx],' and ',frames_list[idx+1],'...'

        # print 'Evaluating frame '+ pr_name
        frame_dir_0 = os.path.join(frames_path, frames_list[idx])
        frame_dir_1 = os.path.join(frames_path, frames_list[idx+1])

        if Backward:
            toExplore_img = cv2.imread(frame_dir_0, 0)
            curr_img = cv2.imread(frame_dir_1, 0)

        else:
            curr_img = cv2.imread(frame_dir_0, 0)
            toExplore_img = cv2.imread(frame_dir_1, 0)

        OF_image = calcOpticalFlowBM(curr_img, toExplore_img, blockSize, areaOfSearch)

        if showResult:
            showOpticalFlowArrows(OF_image, curr_img, frames_list[idx], saveResult=True)
            showOpticalFlowHSVBlockMatching(OF_image, frames_list[idx], visualization='HS', saveResult=True)

        np.save('results/OpticalFlow/files/OF_'+frames_list[idx], OF_image)

    return


def estimateOpticalFlowFarneback(dataset_path, showResult, Backward):

    print 'Optical Flow computation with Farneback method'
    frames_list, gt_list, frames_path, gt_path = readDataset(dataset_path)
    nFrames = len(frames_list)

    for idx in range(0, nFrames, 2):

        print '     --> Analyzing sequence ',frames_list[idx],' and ',frames_list[idx+1],'...'

        # print 'Evaluating frame '+ pr_name
        frame_dir_0 = os.path.join(frames_path, frames_list[idx])
        frame_dir_1 = os.path.join(frames_path, frames_list[idx+1])

        if Backward:
            toExplore_img = cv2.imread(frame_dir_0, 0)
            curr_img = cv2.imread(frame_dir_1, 0)

        else:
            curr_img = cv2.imread(frame_dir_0, 0)
            toExplore_img = cv2.imread(frame_dir_1, 0)

        #OF_image = cv2.calcOpticalFlowFarneback(curr_img, toExplore_img, pyr_scale=0.5, levels=5, winsize=15, iterations=9, poly_n=7, poly_sigma=1.5, flags=0)
        #OF_image = cv2.calcOpticalFlowFarneback(curr_img, toExplore_img, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        OF_image = cv2.calcOpticalFlowFarneback(curr_img, toExplore_img, pyr_scale=0.5, levels=5, winsize=20, iterations=10, poly_n=7, poly_sigma=1.1, flags=0)

        if showResult:
            showOpticalFlowArrows(OF_image, curr_img, frames_list[idx], saveResult = True )
            showOpticalFlowHSVBlockMatching(OF_image, frames_list[idx], visualization='HS', saveResult=True)

        np.save('results/OpticalFlow/files/OF_'+frames_list[idx], OF_image)

    return