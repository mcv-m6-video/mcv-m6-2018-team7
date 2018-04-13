from __future__ import division
import numpy as np
import cv2
from skimage import measure
import ObjectDetected
from estimateSpeed import computeRectificationHomography
import time

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


def drawBoundingBox(img_color, text, top_left, bottom_right, infraction):

    # Get image size
    h, w, d = img_color.shape

    # Get bounding box coord.
    top, left = top_left[1], top_left[0]
    bot, right = bottom_right[1], bottom_right[0]

    # Drawing parameters
    thick = int((h + w) / 300)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.3
    thickness = 1
    if infraction:
        color = [0, 0, 255]
    else:
        color = [0, 255, 0]

    # Draw bounding box
    cv2.rectangle(img_color, (left, top), (right, bot), color, thick)
    size = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img_color, (right - 2, bot - size[0][1] - 4), (right + size[0][0] + 4, bot), color, -1)
    cv2.putText(img_color, text, (right + 2, bot - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return img_color


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


def mergeCloseBoundingBoxes(indexes, detections, frame_number, perspective, h):
    # Check if the bounding boxes objects are too close. Merge them if its the case.
    # This is done in case a car is divided in 2 close CC in the mask.
    merged = False
    merge_dist = 20
    candidate_top_left = (min(indexes[1]), min(indexes[0]))
    candidate_bottom_right = (max(indexes[1]), max(indexes[0]))
    if perspective:
        # If the image is in perspective, do not merge bounding boxes at the top,
        # where cars will be very small, with small separation between each other
        if (candidate_bottom_right[1] + candidate_top_left[1]) / 2 > h / 3:
            for detection in detections:
                min_dist = computeDistanceBetweenBoxes(detection.top_left, detection.bottom_right,
                                                       candidate_top_left, candidate_bottom_right)
                if min_dist < merge_dist:
                    detection.update(frame_number, indexes)
                    merged = True
                    detection.found = True
    else:
        # Else, go ahead and apply the merging to all parts of the image indistinctly
        for detection in detections:
            min_dist = computeDistanceBetweenBoxes(detection.top_left, detection.bottom_right,
                                                   candidate_top_left, candidate_bottom_right)
            if min_dist < merge_dist:
                detection.update(frame_number, indexes)
                merged = True
                detection.found = True

    return merged, detections

def drawStatistics(img_color, text_density, color,  text_avg, text_inf):

    # Get image size
    h, w, d = img_color.shape

    # Drawing parameters
    thick = int((h + w) / 300)
    color_avg = (255, 255, 0)
    color_inf = (0,0,0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1

    # Draw statistics at top-left of the image
    size_density = cv2.getTextSize(text_density, font, scale, thickness)
    size_avg = cv2.getTextSize(text_avg, font, scale, thickness)
    size_inf = cv2.getTextSize(text_inf, font, scale, thickness)

    cv2.rectangle(img_color, (0, 0), (size_density[0][0]+4, size_density[0][1]+4), color, -1)
    cv2.putText(img_color, text_density, (2, size_density[0][1]+2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.rectangle(img_color, (0, size_density[0][1]+4), (size_avg[0][0]+4, size_avg[0][1]+4+size_density[0][1]+4), color_avg, -1)
    cv2.putText(img_color, text_avg, (2, size_density[0][1]+4 + size_avg[0][1]+2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.rectangle(img_color, (0, size_density[0][1]+4 + size_avg[0][1]+4), (size_inf[0][0]+4, size_avg[0][1]+4+size_density[0][1]+4+ size_inf[0][1]+4), color_inf, -1)
    cv2.putText(img_color, text_inf, (2, size_density[0][1]+4 + size_avg[0][1]+4+size_inf[0][1]+2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_color


def paintTrackingResults(input_frame, frame_number, detections, mask, counter, infraction_veh, speed, max_vel=80):

    show_statistics = False

    visible_cars = 0

    frame = input_frame
    for detection in detections:
        if detection.found:
            infraction = False

            if speed:
                if detection.speed > max_vel:
                    infraction = True
                    infraction_veh.append(detection.object_id)
                text = 'ID: ' + str(detection.object_id) + ' ' + ('%.2f' % detection.speed) + 'km/h'
            else:
                text = 'ID: ' + str(detection.object_id)

            frame = drawBoundingBox(frame, text, detection.top_left, detection.bottom_right, infraction)

            visible_cars = visible_cars + 1

    if visible_cars >1:
        if visible_cars>3:
            color = [0, 0, 255]
            text_density = 'HIGH DENSITY: ' + str(visible_cars)
        else:
            color = [0, 255, 255]
            text_density = 'MEDIUM DENSITY: ' + str(visible_cars)
    else:
        color = [0, 255, 0 ]
        text_density = 'LOW DENSITY: ' + str(visible_cars)

    if not infraction_veh:
        n_infractions = 0
    else:
        n_infractions = len(np.unique(infraction_veh))

    avg_density = counter / (((frame_number+1)/30) / 60)
    text_avg = 'Traffic: '+ str(np.round(avg_density))+' cars/min'
    text_inf = 'Infractions: '+str(n_infractions)

    if speed:
        frame = drawStatistics(frame, text_density, color,  text_avg, text_inf)
    result = np.concatenate((frame, np.stack([mask * 255, mask * 255, mask * 255], axis=-1)), 1)
    cv2.imwrite("Results/ComboImage_" + str(frame_number) + '.png', result)

    return result, infraction_veh


def computeTrackingKalman(input_frames, predictions, frame_size, distance_threshold, perspective, speed):

    n_frames, _, d = input_frames.shape
    h, w = frame_size

    # Objects counter + List of detected objects
    counter = 0
    detections = []
    infraction_veh = []

    # Initialize detections using first frame
    frame_number = 0
    frame = input_frames[frame_number, :, :].reshape((h, w, d))
    mask = predictions[frame_number, :].reshape((h, w))
    if speed:
        H, pixelToMeters = computeRectificationHomography(np.uint8(frame))  # for speed estimation
    else:
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        pixelToMeters = 1.0

    start = time.time()

    cc = getConnectedComponents(mask)
    for cc_idx in reversed(range(1, cc.max() + 1)):
        indexes = getLabelCoordinates(cc, cc_idx)

        if not detections:
            # First detection
            counter = counter + 1
            detection = ObjectDetected.ObjectDetected(counter, frame_number, indexes, H, pixelToMeters)
            detection.kalman_filter.updateMeasurement(detection.current_centroid)
            detections.append(detection)
        else:
            merged, detections = mergeCloseBoundingBoxes(indexes, detections, frame_number, perspective, h)
            if not merged:
                # Create new detection if the bounding boxes are far enough
                counter = counter + 1
                detection = ObjectDetected.ObjectDetected(counter, frame_number, indexes, H, pixelToMeters)
                detection.kalman_filter.updateMeasurement(detection.current_centroid)
                detections.append(detection)

    # Paint bounding boxes
    _, infraction_veh = paintTrackingResults(frame, frame_number, detections, mask, counter, infraction_veh, speed)

    # Track the rest of frames
    for frame_number in range(1, n_frames):

        # Analyze new frame. At the moment, no objects have been found
        for detection in detections:
            detection.found = False

        frame = input_frames[frame_number, :, :].reshape((h, w, d))
        mask = predictions[frame_number, :].reshape((h, w))

        cc = getConnectedComponents(mask)
        for cc_idx in reversed(range(1,cc.max()+1)):

            indexes = getLabelCoordinates(cc, cc_idx)

            # Kalman filter
            distances = np.inf * np.ones(len(detections)).astype(float)
            if not detections:
                # If there are no detections in the present,
                # directly create a detection for the current CC
                counter = counter + 1
                detection = ObjectDetected.ObjectDetected(counter, frame_number, indexes, H, pixelToMeters)
                detection.kalman_filter.updateMeasurement(detection.current_centroid)
                detections.append(detection)
            else:
                # If there are detections moving at the present,
                # (1) compute prediction with the Kalman filter of each detection
                # (2) if the centroid of the CC is close enough to some of the predictions,
                #     then the object was found according to Kalman
                for detection_idx in range(0, len(detections)):
                    current_detection = detections[detection_idx]
                    prediction = current_detection.kalman_filter.predictKalmanFilter()
                    centroid = [sum(indexes[0]) / len(indexes[0]), sum(indexes[1]) / len(indexes[1])]
                    distance = current_detection.computeDistance(centroid, prediction)
                    distances[detection_idx] = distance
                    # Paint Kalman prediction for current detection in the frame (green point)
                    if current_detection.found:
                        cv2.circle(frame, (int(prediction[1]), int(prediction[0])), 1, (0, 255, 0), -1)

                min_idx = np.argmin(distances)  # distance from current CC to closest Kalman prediction

                # If the image is in perspective, use smaller distance threshold for cars at the top
                # else, go ahead with the original distance_threshold
                if perspective & (centroid[0] < h / 5):
                    n = 0.4
                else:
                    n = 1.0

                if distances[min_idx] < distance_threshold*n:
                    # Object was found: it is close enough to a Kalman prediction
                    # Update the detection
                    detections[min_idx].update(frame_number, indexes)
                    detections[min_idx].found = True
                else:
                    # Object was not found: it is far from all Kalman predictions
                    # So now we have a new object candidate
                    # We do further checking by using the bounding boxes
                    # If the bounding box of the new object candidate is too close
                    # to an already existing bounding box, merge detections
                    merged, detections = mergeCloseBoundingBoxes(indexes, detections, frame_number, perspective, h)
                    if not merged:
                        # Create new detection if the bounding boxes are far enough
                        counter = counter + 1
                        detection = ObjectDetected.ObjectDetected(counter, frame_number, indexes, H, pixelToMeters)
                        detection.kalman_filter.updateMeasurement(detection.current_centroid)
                        detections.append(detection)

        # Update visibility
        tmp = len(detections)
        detection_idx = 0
        while detection_idx < tmp:
            detection = detections[detection_idx]
            if not detection.found:
                # If current detection was not found and it has been missing for 10 frames, remove it
                if detection.current_frame + 15 < frame_number:
                    detections.remove(detection)
            else:
                # Else, merge close bounding boxes again and end the analysis of the current frame
                for detection2 in detections[detection_idx+1:]:
                    min_dist = computeDistanceBetweenBoxes(detection.top_left, detection.bottom_right,
                                                           detection2.top_left, detection2.bottom_right)
                    if min_dist < 20:
                        if not perspective:
                            detection.update(frame_number, detection2.indexes)
                            detections.remove(detection2)
                        else:
                            if (detection.bottom_right[1] + detection.top_left[1]) / 2 > h / 3:
                                detection.update(frame_number, detection2.indexes)
                                detections.remove(detection2)
            tmp = len(detections)
            detection_idx = detection_idx + 1
        # update counter (for ID assignation)
        if not detections:
            counter = 0
        else:
            counter = detections[-1].object_id

        # Paint bounding boxes
        _, infraction_veh = paintTrackingResults(frame, frame_number, detections, mask, counter, infraction_veh, speed)

    end = time.time()
    fps = n_frames / (end - start)
    print ('FPS for tracking = %.2f' % fps)
    return detections
