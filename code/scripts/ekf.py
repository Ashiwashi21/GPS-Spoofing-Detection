#goal of the script: define an ekf class
#imports
import numpy as np #math

#defines ekf as a resuable object
class ExtendedKalmanFilter:
    #stores all ekf matrices
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F #state transition
        self.H = H #measurement model
        self.Q = Q #process noise
        self.R = R #measurement noise
        self.x0 = x0.copy() #initial state
        self.P0 = P0.copy() #initial covariance

    #reset filter state (resets ekf to initial state before sequence)
    def reset(self):
        self.x = self.x0.copy()
        self.P = self.P0.copy()

    #predicts model (state and covariance, guessing stage)
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    #update step 
    def update(self, z):
        y = z - self.H @ self.x #residual
        S = self.H @ self.P @ self.H.T + self.R #innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) #kalman gain
        self.x = self.x + K @ y #update state estimate
        self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P #update covariance
        return y #retuns residual (used for spoofing detection)

    #run ekf on a full sequence (reset, predict, store)
    def run(self, sequence):
        self.reset()
        residuals = []
        for z in sequence:
            self.predict()
            y = self.update(z)
            residuals.append(y)
        return np.array(residuals)
