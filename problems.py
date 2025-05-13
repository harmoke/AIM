import numpy as np
from abc import ABC, abstractmethod

class Problem(ABC):
    @abstractmethod
    def loss(self, w, **params): pass
    
    @abstractmethod
    def gradient(self, w, **params): pass

class LogisticRegressionL2(Problem):
    def __init__(self, A, b, l = 1.0):
        self.A = A
        self.b = b
        self.l = l
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, x):
        z = self.A @ x
        logistic_loss = np.mean((1 - self.b) * z + np.log(1 + np.exp(-z)))
        regularization = 0.5 * self.l * np.sum(x**2)
        return logistic_loss + regularization
    
    def gradient(self, x):
        pred = self.sigmoid(self.A @ x)
        grad = self.A.T @ (pred - self.b) / len(self.b)
        grad += self.l * x
        return grad

class SmoothedLpL2Problem:
    def __init__(self, A, b, epsilon=1e-1, p=0.5):
        self.A = A
        self.b = b
        self.l = 0.2 * np.linalg.norm(A.T @ b, np.inf)
        self.epsilon = epsilon
        self.p = p
        self.m, self.n = A.shape
        
    def _s(self, x):
        return np.where(
            np.abs(x) > self.epsilon,
            np.abs(x),
            (x**2)/(2*self.epsilon) + self.epsilon/2
        )
    
    def _ds_dx(self, x):
        return np.where(
            np.abs(x) > self.epsilon,
            np.sign(x),
            x/self.epsilon
        )
    
    def loss(self, x):
        residual = self.A @ x - self.b
        l2_term = 0.5 * np.sum(residual**2)
        
        s_values = self._s(x)
        lp_term = self.l * np.sum(s_values**self.p)
        
        return l2_term + lp_term
    
    def gradient(self, x):
        residual = self.A @ x - self.b
        grad_l2 = self.A.T @ residual
        
        s_values = self._s(x)
        ds_dx = self._ds_dx(x)
        grad_lp = self.l* self.p * (s_values**(self.p - 1)) * ds_dx
        
        return grad_l2 + grad_lp