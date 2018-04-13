from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from skimage.morphology import remove_small_objects, binary_closing, disk, rectangle
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.metrics import precision_recall_fscore_support
import cv2

def evaluateBackgroundEstimation(predictions, gts):

    predictions = np.reshape(predictions, gts.shape[0]*gts.shape[1])
    gts = np.reshape(gts, gts.shape[0] * gts.shape[1])

    #Remove array positions where gt >=85 & gt<=170
    predictions = predictions[(gts != -1)]
    gts = gts[(gts != -1)]

    precision, recall, fscore, support = precision_recall_fscore_support(gts.astype(int), predictions.astype(int), average = 'binary')
    #print 'F1-Score = ',fscore
    return precision, recall, fscore

def updateGaussians(pred, frame, mu_vec, sigma_vec, rho):

    mu_vec[(pred==0) & (frame!=-1)] = (rho * frame[(pred==0) & (frame!=-1)]) + (1-rho) * mu_vec[(pred==0) & (frame!=-1)]
    sigma_vec[(pred==0) & (frame!=-1)] = np.sqrt(rho*np.power(frame[(pred==0) & (frame!=-1)]-mu_vec[(pred==0) & (frame!=-1)],2) + (1-rho)*np.power(sigma_vec[(pred==0) & (frame!=-1)],2))

    return mu_vec, sigma_vec

class AdaptiveClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha='alpha', rho='rho', mu='mu', std ='std', mu_c='mu_c', std_c='std_c'):
        self.alpha = alpha
        self.rho = rho
        self.mu = mu
        self.std = std
        self.mu_c = mu_c
        self.std_c = std_c

    def fit(self, X, y):

        mu = np.zeros([1,X.shape[1]])
        std = np.zeros([1, X.shape[1]])
        
        for i in range(X.shape[1]):
            all_pixels = np.reshape(X[:,i], (X.shape[0],1))
            if sum((all_pixels==-1).astype(int)) == X.shape[0]:
                mu[0,i] = 0
                std[0,i] = 0
            else:
                mu[0,i] = np.mean(all_pixels[all_pixels!=-1], 0)
                std[0,i] = np.std(all_pixels[all_pixels!=-1], 0)
        self.mu = mu.reshape((1, X.shape[1]))
        self.std = std.reshape((1, X.shape[1]))

        return self

    def predict(self, X, y):
        predictions = np.zeros([1, X.shape[1]])
        for idx in range(len(X)):

            current_frame = X[idx, :].reshape((1, X.shape[1]))
            prediction = (abs(current_frame - self.mu) >= self.alpha * (self.std + 2)).astype(int)
            prediction[current_frame==-1] = 0
            predictions = np.vstack((predictions, prediction))

            # Update step
            updated_mu, updated_std = updateGaussians(prediction,current_frame,self.mu,self.std,self.rho)
            self.mu = updated_mu
            self.std = updated_std

        predictions = predictions[1:, :]
        return predictions

    def postProcessing(self, predictions, frame_size, holeFilling, areaFiltering, P, connectivity, Morph):

        nFrames, nPixels = predictions.shape

        if connectivity == 8:
            se = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # 8-connectivity
        else:
            se = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]  # 4-connectivity

        for frame in range(0, nFrames):
            actualFrame = np.reshape(predictions[frame, :], (frame_size[0], frame_size[1]))

            if holeFilling:
                #cv2.imwrite('Before_holefilling.png', 255 * actualFrame)
                actualFrame = binary_fill_holes(actualFrame.astype(int), structure=se)
                #cv2.imwrite('After_holefilling.png', 255 * actualFrame)

            if areaFiltering:
                actualFrame = remove_small_objects(actualFrame, P)

            if Morph:
                # SE = disk(2,2)
                SE = rectangle(4,2)
            
                #cv2.imwrite('Before_closing.png', 255 * actualFrame)
                actualFrame = binary_closing(actualFrame.astype(int), selem=SE, out=None)
                # cv2.imwrite('Results/After_closing'+str(frame)+'.png', 255 * actualFrame)

                actualFrame = binary_fill_holes(actualFrame.astype(int), structure=se)

            predictions[frame, :] = np.reshape(actualFrame.astype(int), (1, frame_size[0] * frame_size[1]))
        return predictions

    def score(self, X, y):

        predictions = self.predict(X, y)
        precision, recall, fscore = evaluateBackgroundEstimation(predictions, y)

        return fscore

    def performance_measures(self, X, y, frame_size,  holeFilling, areaFiltering, P, connectivity, Morph):

        predictions = self.predict(X, y)
        predictions = self.postProcessing(predictions, frame_size, holeFilling, areaFiltering, P, connectivity, Morph)
        precision, recall, fscore = evaluateBackgroundEstimation(predictions, y)


        return precision, recall, fscore
