import numpy as np


class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr
        self.weights = None
        self.bias = None

    def fit(self, X: np.array, y: np.array):
        m = 0
        c = 0
        L = self.lr
        n = float(len(X))
        
        for i in range(self.epochs):
            y_pred = m*X + c
            residuals = y_pred - y
            loss = np.sum(residuals ** 2)
            
            D_m = (-2/n) * sum(X * residuals)
            D_c = (-2/n) * sum(residuals)
            
            m = m - L * D_m
            c = c - L * D_c
            
        self.weights = m
        self.bias = c
    

    def predict(self, X):
        y_pred = self.weights*X + self.bias
        return y_pred