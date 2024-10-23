import numpy as np
from model import model

class LinearRegression(model):
    def __init__(self):
        self.V_m = None
        self.V_n = None
        self.m = None
        self.n = None
        return self
    
    def weighted_average(Z, Zerr):
        return (Z/Zerr**2).sum()/(1/Zerr**2).sum()
    
    def fit(self, X, Y, Yerr):
        self.y_wa = self.weighted_average(Y, Yerr)
        self.xy_wa = self.weighted_average(X*Y, Yerr)
        self.x_wa = self.weighted_average(X, Yerr)
        self.x_sq_wa = self.weighted_average(X**2, Yerr)
        
        self.sigma_wa = Yerr.size/(1/Yerr**2).sum()
        self.V_m = self.sigma_wa/(X.size*(self.x_sq_wa-self.x_wa**2))
        self.V_n = sigma_wa/(X.size*(self.x_sq_wa-self.x_wa**2))*self.x_sq_wa
        self.m = (self.xy_wa-self.x_wa*self.y_wa)/(self.x_sq_wa-self.x_wa**2)
        self.n = (self.x_sq_wa*self.y_wa-self.x_wa*self.xy_wa)/(self.x_sq_wa-self.x_wa**2)
        
        return self
    
    def setParameters(self, m, V_m, n, V_n):
        self.m = m
        self.n = n
        self.V_m = V_m
        self.V_n = V_n
        
    def getParameters(self):
        return {"m": self.m, "V_m": self.V_m, "n": self.n, "V_n": self.V_n}
    
    def predict(self, x):
        return x*self.m+self.n
    
    def predict_most_and_least(self, x):
        return [x*(self.m+self.V_m**(1/2))+self.n+self.V_n**(1/2), x*(self.m-self.V_m**(1/2))+self.n-self.V_n**(1/2)]
    