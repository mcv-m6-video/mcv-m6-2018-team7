import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from backgroundEstimation import evaluateBackgroundEstimation

def updateGaussians(pred, frame, mu_vec, sigma_vec, rho):

    mu_vec[pred==0] = (rho * frame[pred==0]) + (1-rho) * mu_vec[pred==0]
    sigma_vec[pred==0] = np.sqrt(rho*np.power(frame[pred==0]-mu_vec[pred==0],2) + (1-rho)*np.power(sigma_vec[pred==0],2))

    return mu_vec, sigma_vec

class AdaptiveClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha='alpha', rho='rho', mu='mu', std ='std'):
        self.alpha = alpha
        self.rho = rho
        self.mu = mu
        self.std = std

    def fit(self, X, y):

        mu = np.mean(X, 0)
        std = np.std(X, 0)
        self.mu = mu.reshape((1, X.shape[1]))
        self.std = std.reshape((1, X.shape[1]))

        return self

    def predict(self, X, y):
        predictions = np.zeros([1, X.shape[1]])
        for idx in range(len(X)):

            current_frame = X[idx, :].reshape((1, X.shape[1]))
            prediction = (abs(current_frame - self.mu) >= self.alpha * (self.std + 2)).astype(int)
            predictions = np.vstack((predictions, prediction))

            # Update step
            updated_mu, updated_std = updateGaussians(prediction,current_frame,self.mu,self.std,self.rho)
            self.mu = updated_mu
            self.std = updated_std

        predictions = predictions[1:, :]
        return predictions

    def score(self, X, y):

        predictions = self.predict(X, y)
        precision, recall, fscore = evaluateBackgroundEstimation(predictions, y)

        return fscore

    def performance_measures(self, X, y):

        predictions = self.predict(X, y)
        precision, recall, fscore = evaluateBackgroundEstimation(predictions, y)

        return precision, recall, fscore
