import numpy as np
import cv2
import os
from AdaptiveClassifier import AdaptiveClassifier
from readInputWriteOutput import readDataset
import utilsTracking as utils
import trackingKalman as kalman
import shutil
import time


def testKalmanTracking(dataset_path, ID, alpha, rho, holeFilling, areaFiltering, P, connectivity, Morph, GT_available, estimate_speed=False):

    # read dataset
    #train_rgb, test_rgb, train_gray, test_gray, train_gt, test_gt, frame_size  = readDatasetStabilized(dataset_path, ID, True, GT_available)
    train_rgb, test_rgb, train_gray, test_gray, train_gt, test_gt, frame_size = readDataset(dataset_path, ID, True, GT_available)

    # compute masks
    init_mu = np.zeros([1, train_gray.shape[1]])
    init_std = np.zeros([1, train_gray.shape[1]])
    init_mu_c = np.zeros([1, train_gray.shape[1], 3])
    init_std_c = np.zeros([1, train_gray.shape[1], 3])
    clf = AdaptiveClassifier(alpha, rho, init_mu, init_std, init_mu_c, init_std_c)
    clf = clf.fit(train_gray, train_gt)
    predictions = clf.predict(test_gray, test_gt)
    predictions = clf.postProcessing(predictions, frame_size, holeFilling, areaFiltering, P, connectivity, Morph)

    # tracking starts here
    distance_threshold = 40
    object_list = kalman.computeTrackingKalman(test_rgb, predictions, frame_size, distance_threshold, True,
                                                 speed=estimate_speed)



# Available trackers in OpenCV
# tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerTLD_create()
# tracker = cv2.TrackerMedianFlow_create()
# tracker = cv2.TrackerGOTURN_create()
def testMedianFlowTracking(dataset_path, ID, alpha, rho, holeFilling, areaFiltering, P, connectivity, Morph, GT_available):

    # read dataset
    train_rgb, test_rgb, train_gray, test_gray, train_gt, test_gt, frame_size = readDataset(dataset_path, ID, True, GT_available)

    # compute masks
    init_mu = np.zeros([1, train_gray.shape[1]])
    init_std = np.zeros([1, train_gray.shape[1]])
    init_mu_c = np.zeros([1, train_gray.shape[1], 3])
    init_std_c = np.zeros([1, train_gray.shape[1], 3])
    clf = AdaptiveClassifier(alpha, rho, init_mu, init_std, init_mu_c, init_std_c)
    clf = clf.fit(train_gray, train_gt)
    predictions = clf.predict(test_gray, test_gt)
    predictions = clf.postProcessing(predictions, frame_size, holeFilling, areaFiltering, P, connectivity, Morph)

    n_frames, _, d = test_rgb.shape
    h, w = frame_size

    os.mkdir(dataset_path + '/' + ID + '/tmp')
    for frame_number in range(0, n_frames):
        frame = np.reshape(test_rgb[frame_number, :, :], (h, w, 3))
        cv2.imwrite(dataset_path + '/' + ID + '/tmp/tmp_'+('%03d' % frame_number)+'.png', frame)

    # tracking starts here
    frame_num = 0
    prev_bbox = []
    tracker_list = []
    perspective = True
    idx_removed = []
    num_trackers = 0
    video = cv2.VideoCapture(dataset_path+'/'+ID+'/tmp/tmp_%03d.png')

    start = time.time()

    for frame_number in range(0, n_frames):

        ok, frame = video.read()
        mask = np.reshape(predictions[frame_number, :], (h, w))

        # Initialize new trackers if necessary
        tracker_list, num_new_trackers = utils.initialize_trackers(frame, mask, tracker_list, prev_bbox, perspective)
        if num_new_trackers > 0:
            num_trackers = num_trackers + num_new_trackers

        # Update trackers
        bbox_list = []
        for i in range(np.shape(tracker_list)[0]):
            ok, bbox = tracker_list[i].update(frame)
            bbox_list.append(bbox)

        # Check if the predicted bboxes (bbox_list) overlap with a CC
        # if they do not overlap, then they must be removed
        idx_removed = utils.check_overlap_out(bbox_list, mask, perspective)
        tmp = np.shape(bbox_list)[0]
        idx = 0
        while idx < tmp:
            if idx in idx_removed:
                del tracker_list[idx]
                del bbox_list[idx]
                tmp = tmp - 1
            idx = idx + 1

        # update bboxes
        prev_bbox = bbox_list

        car_ID_list = range(1, num_trackers + 1)
        num_cars = utils.count_cars(bbox_list)  # currently on frame

        # Draw bounding boxes
        for i in range(np.shape(bbox_list)[0]):
            p1 = (int((bbox_list[i])[0]), int((bbox_list[i])[1]))
            p2 = (int((bbox_list[i])[0] + (bbox_list[i])[2]), int((bbox_list[i])[1] + (bbox_list[i])[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.rectangle(frame, (p2[0] + 2, p2[1] - 18), (p2[0] + 38, p2[1] - 2), (0, 0, 0), -1, 1)
            cv2.putText(frame, 'ID: ' + str(car_ID_list[i]), (p2[0] + 5, p2[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)

        # Display result
        cv2.putText(frame, 'Number of cars: ' + str(num_cars), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # cv2.imshow('Tracking', frame)
        result = np.concatenate((frame, np.stack([mask * 255, mask * 255, mask * 255], axis=-1)), 1)
        cv2.imwrite("Results/ComboImage_" + str(frame_num) + '.png', result)
        frame_num = frame_num + 1

    shutil.rmtree(dataset_path+'/'+ID+'/tmp')

    end = time.time()
    fps = n_frames / (end - start)
    print ('FPS for tracking = %.2f' % fps)