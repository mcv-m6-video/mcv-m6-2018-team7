from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk, rectangle
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
                actualFrame = binary_fill_holes(actualFrame.astype(int), structure = se)
                #cv2.imwrite('AFter_holefilling.png', 255 * actualFrame)

            if areaFiltering:
                actualFrame = remove_small_objects(actualFrame, P)

            if Morph:
                # SE = disk(2,2)
                SE = rectangle(4,2)
            
                #cv2.imwrite('Before_closing.png', 255 * actualFrame)
                actualFrame = binary_closing(actualFrame.astype(int), selem=SE, out=None)
                cv2.imwrite('Stabilization/After_closing'+str(frame)+'.png', 255 * actualFrame)

                actualFrame = binary_fill_holes(actualFrame.astype(int), structure = se)

            predictions[frame, :] = np.reshape(actualFrame.astype(int), (1, frame_size[0] * frame_size[1]))
        return predictions


    def shadowRemoval(self, test_frames_c, predictions, frame_size):

        for idx in range(len(predictions)):
            current_frame = test_frames_c[idx, :, :].reshape((1, predictions.shape[1],3))
            current_prediction = predictions[idx,:].reshape((1, predictions.shape[1]))

            # method1 -> HSV and thresholds based on relationship with respect to background image
            #            Inspired on Cucchiara, R., Grana C., et al. *Improving Shadow Suppression
            #            in Moving Object Detection with HSV Color Information* (2001).
            #            Fundamental ideas:
            #            - Value of shadowed areas should be significantly lower than mean value
            #            - Hue and Saturation of shadowed areas should be similar to mean values
            #shadow_mask = ((0.0 <= (current_frame[:, :, 2] / self.mu_c[:, :, 2]))
            #                 & ((current_frame[:,:,2] / self.mu_c[:,:,2]) <= 0.6)
            #                 & (abs(current_frame[:, :, 0]/255. - self.mu_c[:, :, 0]/255.) <= 0.1)
            #                 & (abs(current_frame[:, :, 1]/255. - self.mu_c[:, :, 1]/255.) <= 0.1)
            #                 & (current_prediction == 1) )

            # method2 -> Learn shadows from ground truth and model their HSV values as a Guassian Distribution
            #            Shadow values should ideally fall in a range of type [mu-th*sigma, mu+th*sigma]
            #            th is a deviation threshold (th = 2 guarantees 95% of shadows will be detected)
            #            Drawback: certain parts of the cars are very similar to shadowed road,
            #                      if th is excessively permissive then recall decreases too much
            muH, muS, muV = 108.016, 51.169, 24.396
            sigmaH, sigmaS, sigmaV = 46.798, 30.513, 14.323
            th = 0.5 # deviation threshold
            shadow_mask = ( (current_frame[:, :, 0] >= muH - th * sigmaH) & (current_frame[:, :, 0] <= muH + th * sigmaH) &
                            (current_frame[:, :, 1] >= muS - th * sigmaS) & (current_frame[:, :, 1] <= muS + th * sigmaS) &
                            (current_frame[:, :, 2] >= muV - th * sigmaV) & (current_frame[:, :, 2] <= muV + th * sigmaV) &
                            (current_prediction == 1) )

            final_prediction = current_prediction - shadow_mask.astype(int)

            # save 3 images (1) input detection (2) input detection with detected shadows in red
            #               (3) input detection without shadows
            # input = current_prediction
            # tmp = np.zeros([frame_size[0], frame_size[1], 3])
            # current_prediction[shadow_mask] = 0
            # tmp[:, :, 0] = np.reshape(current_prediction, (frame_size[0], frame_size[1]))
            # tmp[:, :, 1] = np.reshape(current_prediction, (frame_size[0], frame_size[1]))
            # current_prediction[shadow_mask] = 1
            # tmp[:, :, 2] = np.reshape(current_prediction, (frame_size[0], frame_size[1]))
            # cv2.imwrite('shadow_detection/' + str(idx) + '_1_input.png', 255 * np.reshape(input, (frame_size[0], frame_size[1])))
            # cv2.imwrite('shadow_detection/' + str(idx) + '_2_shadows.png', 255 * tmp)
            # cv2.imwrite('shadow_detection/' + str(idx) + '_3_final_prediction.png', 255 * np.reshape(final_prediction, (frame_size[0], frame_size[1])))

            predictions[idx, :] = final_prediction

        return predictions

    def fit_shadowRemoval(self, train_frames_c):

        mu_c = np.mean(train_frames_c, 0)
        std_c = np.std(train_frames_c, 0)
        self.mu_c = mu_c.reshape((1, train_frames_c.shape[1], 3))
        self.std_c = std_c.reshape((1, train_frames_c.shape[1], 3))

        return self

    def score(self, X, y):

        predictions = self.predict(X, y)
        precision, recall, fscore = evaluateBackgroundEstimation(predictions, y)

        return fscore

    def performance_measures(self, X, y, frame_size,  holeFilling, areaFiltering, P, connectivity, Morph):

        predictions = self.predict(X, y)
        predictions = self.postProcessing(predictions, frame_size, holeFilling, areaFiltering, P, connectivity, Morph)
        precision, recall, fscore = evaluateBackgroundEstimation(predictions, y)

        return precision, recall, fscore

    def performance_measures_shadowRemoval(self, train_frames_c, test_frames_bw, test_frames_c, test_gt, frame_size,  holeFilling, areaFiltering, P, connectivity, Morph):

        self.fit_shadowRemoval(train_frames_c)
        predictions = self.predict(test_frames_bw, test_gt)
        predictions = self.postProcessing(predictions, frame_size, holeFilling, areaFiltering, P, connectivity, Morph)
        predictions = self.shadowRemoval(test_frames_c, predictions, frame_size)
        precision, recall, fscore = evaluateBackgroundEstimation(predictions, test_gt)

        return precision, recall, fscore
