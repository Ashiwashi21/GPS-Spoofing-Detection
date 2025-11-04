import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x0 = x0.copy()
        self.P0 = P0.copy()

    def reset(self):
        self.x = self.x0.copy()
        self.P = self.P0.copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P
        return y

    def run(self, sequence):
        self.reset()
        residuals = []
        for z in sequence:
            self.predict()
            y = self.update(z)
            residuals.append(y)
        return np.array(residuals)
