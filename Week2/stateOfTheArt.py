import cv2
import numpy as np
import os


def backgroundSubtractorMOG(dataset_path, ID, method, test_frames):

    #Background estimation
    if method == 'MOG':
        print 'Running background substraction using MOG ...'
        fgbg = cv2.BackgroundSubtractorMOG()
        if not os.path.exists(dataset_path+ID+'/MOG'):
            os.makedirs(dataset_path+ID+'/MOG')
    elif method == 'MOG2':
        print 'Running background substraction using MOG2 ...'
        fgbg = cv2.BackgroundSubtractorMOG2()
        if not os.path.exists(dataset_path+ID+'/MOG2'):
            os.makedirs(dataset_path+ID+'/MOG2')

    frames_path = dataset_path + ID + '/input'
    frame_list = sorted(os.listdir(frames_path))
    tmp = cv2.imread(os.path.join(frames_path, frame_list[0]))
    mask_vec = np.zeros([1,tmp.shape[0]*tmp.shape[1]])

    for idx in range(len(test_frames)):
        im=np.reshape(test_frames[idx],(tmp.shape[0],tmp.shape[1]))
        fgmask = fgbg.apply(im, learningRate=0.05)
        fgmask = fgmask.ravel()
        fgmask[fgmask == 127] = 0
        fgmask[fgmask > 127] = 1
        mask_vec = np.vstack((mask_vec,fgmask))
        cv2.imwrite(dataset_path+ID+'/'+method+'/img_'+str(idx)+'.png', np.reshape(fgmask*255,(tmp.shape[0],tmp.shape[1])))

    mask_vec = mask_vec[1:,:]
    predictions = mask_vec
    return predictions