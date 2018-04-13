import numpy as np
import KalmanFilter as kf
from estimateSpeed import estimate_speed

class ObjectDetected:

    def __init__(self, object_id, frame_number, indexes, H, pixelToMeters):
        self.object_id = object_id
        self.indexes = indexes
        self.current_frame = frame_number
        self.frames = [self.current_frame]
        self.top_left = (min(self.indexes[1]), min(self.indexes[0]))
        self.bottom_right = (max(self.indexes[1]), max(self.indexes[0]))
        self.width = self.bottom_right[0] - self.top_left[0]
        self.height = self.bottom_right[1] - self.top_left[1]
        self.current_centroid = (sum(self.indexes[0])/len(self.indexes[0]),
                                 sum(self.indexes[1])/len(self.indexes[1]))
        self.centroids = [self.current_centroid]
        self.kalman_filter = kf.KalmanFilter(self.object_id, self.current_frame, self.current_centroid)
        self.found = True
        self.speed = 40.0
        self.speeds = [self.speed]
        self.H = H
        self.pixelToMeters = pixelToMeters

    def update(self, frame_number, indexes):

        if frame_number == self.current_frame:
            updated_indexes = (np.concatenate((self.indexes[0], indexes[0]), axis=0),
                               np.concatenate((self.indexes[1], indexes[1]), axis=0))
            self.indexes = updated_indexes
            self.top_left = (min(self.indexes[1]), min(self.indexes[0]))
            self.bottom_right = (max(self.indexes[1]), max(self.indexes[0]))
            self.width = self.bottom_right[0] - self.top_left[0]
            self.height = self.bottom_right[1] - self.top_left[1]
            self.current_centroid = (sum(self.indexes[0]) / len(self.indexes[0]),
                                     sum(self.indexes[1]) / len(self.indexes[1]))
            self.centroids[-1] = self.current_centroid
            self.found = True
        else:
            self.current_frame = frame_number
            self.frames.append(self.current_frame)
            self.indexes = indexes
            self.top_left = (min(indexes[1]), min(indexes[0]))
            self.bottom_right = (max(indexes[1]), max(indexes[0]))
            self.width = self.bottom_right[0] - self.top_left[0]
            self.height = self.bottom_right[1] - self.top_left[1]
            self.current_centroid = sum(indexes[0]) / len(indexes[0]), sum(indexes[1]) / len(indexes[1])
            self.centroids.append(self.current_centroid)
            self.kalman_filter.updateMeasurement(self.current_centroid)
            if (frame_number % 3 == 0) & (frame_number > 3):
                actual_speed = estimate_speed(self.current_centroid, self.current_frame,
                                              self.centroids[len(self.centroids) - 4],
                                              self.frames[len(self.frames) - 4], H=self.H, fps=25,
                                              PixelToMeters=self.pixelToMeters)
                self.speed = actual_speed * 0.33 + self.speeds[len(self.speeds) - 1] * 0.33 + self.speeds[len(self.speeds) - 2] * 0.33 + 40
            self.found = True

    def computeDistance(self, point1, point2):
        distance = pow((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2, 0.5)
        return distance
