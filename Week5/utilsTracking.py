import numpy as np
from skimage import measure
import cv2

def getConnectedComponents(img_mask):
    # Binarize the image
    if img_mask.max()> 1:
        img_mask = np.where(img_mask > 1, 1, 0)
    # Detect connected components and assign an ID to each one
    connected_components_img = measure.label(img_mask, background=0)
    return connected_components_img


def getLabelCoordinates(connected_components_img, idx):
    # Find where labels in connected_components_img are equal to idx
    return np.where(connected_components_img == idx)


def obtain_bbox(car_idx, img_mask):
    connected_components_img = getConnectedComponents(img_mask)
    indexes = getLabelCoordinates(connected_components_img, car_idx)
    top_left = (min(indexes[1]), min(indexes[0]))
    bottom_right = (max(indexes[1]), max(indexes[0]))
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]
    bbox = (top_left[0], top_left[1], height, width)
    return bbox


def count_cars(bbox_list):
    cont = 0
    for i in range(np.shape(bbox_list)[0]):
        if sum(bbox_list[i]) > 0:
            cont = cont+1
    return cont


def computeDistanceBetweenBoxes(top_left1, bottom_right1, top_left2, bottom_right2):

    def dist(point1, point2):
        distance = pow((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2, 0.5)
        return distance

    x1, y1 = top_left1[1], top_left1[0]
    x1b, y1b = bottom_right1[1], bottom_right1[0]

    x2, y2 = top_left2[1], top_left2[0]
    x2b, y2b = bottom_right2[1], bottom_right2[0]

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0


def mergeCloseBoundingBoxes(candidate_bbox, idx, bbox_list, perspective, h):
    # Check if the bounding boxes objects are too close. Merge them if its the case.
    # This is done in case a car is divided in 2 close CC in the mask.
    merged = False
    candidate_top_left = (candidate_bbox[0], candidate_bbox[1])
    candidate_bottom_right = (candidate_bbox[0] + candidate_bbox[2], candidate_bbox[1] + candidate_bbox[3])
    if perspective:
        # If the image is in perspective, do not merge bounding boxes at the top,
        # where cars will be very small, with small separation between each other
        if (candidate_bottom_right[1] + candidate_top_left[1]) / 2 > h / 3:
            for i in range(np.shape(bbox_list)[0]):
                current_bbox = bbox_list[i]
                bbox_top_left = (current_bbox[0], current_bbox[1])
                bbox_bottom_right = (current_bbox[0]+current_bbox[2], current_bbox[1]+current_bbox[3])

                min_dist = computeDistanceBetweenBoxes(bbox_top_left, bbox_bottom_right,
                                                       candidate_top_left, candidate_bottom_right)
                if (min_dist < 20) & (i <> idx):
                    top = min(bbox_top_left[0], candidate_top_left[0])
                    left = min(bbox_top_left[1], candidate_top_left[1])
                    bottom = max(bbox_bottom_right[0], candidate_bottom_right[0])
                    right = max(bbox_bottom_right[1], candidate_bottom_right[1])
                    width = right - left
                    height = bottom - top
                    merged_bbox = (top, left, height, width)
                    bbox_list[i] = merged_bbox
                    merged = True
    else:
        # Else, go ahead and apply the merging to all parts of the image indistinctly
        for i in range(np.shape(bbox_list)[0]):
            current_bbox = bbox_list[i]
            bbox_top_left = (current_bbox[0], current_bbox[1])
            bbox_bottom_right = (current_bbox[0] + current_bbox[2], current_bbox[1] + current_bbox[3])

            min_dist = computeDistanceBetweenBoxes(bbox_top_left, bbox_bottom_right,
                                                   candidate_top_left, candidate_bottom_right)
            if (min_dist < 20) & (i <> idx):
                top = min(bbox_top_left[0], candidate_top_left[0])
                left = min(bbox_top_left[1], candidate_top_left[1])
                bottom = max(bbox_bottom_right[0], candidate_bottom_right[0])
                right = max(bbox_bottom_right[1], candidate_bottom_right[1])
                width = right - left
                height = bottom - top
                merged_bbox = (top, left, height, width)
                bbox_list[i] = merged_bbox
                merged = True

    return merged, bbox_list


def requires_new_tracker(candidate_bbox, prev_bbox):

    min_dist = np.inf
    top_left1 = (candidate_bbox[0], candidate_bbox[1])
    bottom_right1 = (candidate_bbox[0] + candidate_bbox[2], candidate_bbox[1] + candidate_bbox[3])
    for j in range(np.shape(prev_bbox)[0]):
        bbox2 = prev_bbox[j]

        top_left2 = (bbox2[0], bbox2[1])
        bottom_right2 = (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
        minimum = computeDistanceBetweenBoxes(top_left1, bottom_right1, top_left2, bottom_right2)

        if minimum < min_dist:
            min_dist = minimum

    if min_dist > 10:
        return True

    return False


# Merge close connected components
def merge_bbox(bbox_list, perspective, h):
    tmp = np.shape(bbox_list)[0]
    idx = 0
    while idx < tmp:
        current_bbox = bbox_list[idx]
        merged, bbox_list = mergeCloseBoundingBoxes(current_bbox, idx, bbox_list, perspective, h)
        if merged:
            del bbox_list[idx]
            tmp = tmp - 1
        idx = idx + 1
    return bbox_list


def initialize_trackers(frame, img_mask, tracker_list, prev_bbox, perspective):
    num_new_trackers = 0
    # Initialize trackers on the first frame
    if not prev_bbox:
        cc = getConnectedComponents(img_mask)
        bbox_list = []
        for i in range(max(cc.flatten())):
            bbox = obtain_bbox(i + 1, img_mask)
            bbox_list.append(bbox)

        # Merge bbox
        bbox_list = merge_bbox(bbox_list, perspective, img_mask.shape[0])

        for i in range(np.shape(bbox_list)[0]):
            tracker = cv2.TrackerMedianFlow_create()
            tracker.init(frame, bbox_list[i])
            tracker_list.append(tracker)
            num_new_trackers = num_new_trackers + 1
    else:
        candidate_bbox_list = []
        cc = getConnectedComponents(img_mask)
        for i in range(max(cc.flatten())):
            candidate_bbox = obtain_bbox(i + 1, img_mask)
            candidate_bbox_list.append(candidate_bbox)

        # Merge candidate bboxes
        candidate_bbox_list = merge_bbox(candidate_bbox_list, perspective, img_mask.shape[0])
        # Check overlap and create new trackers if necessary
        for i in range(np.shape(candidate_bbox_list)[0]):
            candidate_bbox = candidate_bbox_list[i]
            if requires_new_tracker(candidate_bbox, prev_bbox):
                tracker = cv2.TrackerMedianFlow_create()
                tracker.init(frame, candidate_bbox)
                tracker_list.append(tracker)
                num_new_trackers = num_new_trackers + 1

    return tracker_list, num_new_trackers


def check_overlap_out(pred_bbox_list, img_mask, perspective):
    candidate_bbox_list = []
    idx_removed = []
    cc = getConnectedComponents(img_mask)
    for i in range(max(cc.flatten())):
        candidate_bbox = obtain_bbox(i + 1, img_mask)
        candidate_bbox_list.append(candidate_bbox)

    # Merge candidate bboxes
    candidate_bbox_list = merge_bbox(candidate_bbox_list, perspective, img_mask.shape[0])
    # Check overlap with predicted bboxes. If no overlap, remove the corresponding predicted bbox
    tmp = np.shape(pred_bbox_list)[0]
    idx = 0
    while idx < tmp:
        current_bbox = pred_bbox_list[idx]
        if requires_new_tracker(current_bbox, candidate_bbox_list):
            #del pred_bbox_list[idx]
            idx_removed.append(idx)
            tmp = tmp - 1
        idx = idx + 1

    return idx_removed
