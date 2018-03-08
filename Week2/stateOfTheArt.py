import cv2
import numpy as np
import os
from readInputWriteOutput import readDataset
from backgroundEstimation import evaluateBackgroundEstimation

def testBackgroundSubtractorMOG(dataset_path, ID, method, save_masks, Color):

    # Check directories
    output_dir = dataset_path+ID+'/'+method
    print 'Running background subtraction using '+method+' ...'
    print 'Output masks will be saved at "' + output_dir + '" directory'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use MOG or MOG2 according to input method
    if method == 'MOG':
        fgbg = cv2.BackgroundSubtractorMOG()
    elif method == 'MOG2':
        fgbg = cv2.BackgroundSubtractorMOG2()


    # Get input frames size
    frames_path = dataset_path + ID + '/input'
    frame_list = sorted(os.listdir(frames_path))
    nrows, ncols, nchannels = cv2.imread(os.path.join(frames_path, frame_list[0])).shape
    mask_vec = np.zeros([1,nrows*ncols])

    # Read dataset
    train_frames, test_frames, train_gts, test_gts = readDataset(dataset_path, ID, Color)

    # Run MOG
    for idx in range(len(test_frames)):
        if Color:
            im = np.uint8( np.reshape(test_frames[idx], (nrows,ncols,nchannels)) )
        else:
            im = np.uint8( np.reshape(test_frames[idx], (nrows, ncols)) )
        fgmask = fgbg.apply(im, learningRate=0.01)
        fgmask = fgmask.ravel()
        fgmask[fgmask == 127] = 0
        fgmask[fgmask > 127] = 1
        mask_vec = np.vstack((mask_vec,fgmask))
        # Save current mask
        if save_masks:
            cv2.imwrite(output_dir+'/img_'+('%03d' % idx)+'.png', np.reshape(255*fgmask,(nrows,ncols)))

    # Get predictions and compute performance measures
    predictions = mask_vec[1:,:]
    precision, recall, fscore = evaluateBackgroundEstimation(predictions, test_gts)
    print 'F-score:', fscore
    print 'Precision: ', precision
    print 'Recall: ', recall


