import pdfkit
from csv2pdf import convert
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

class model:
    def __init__(self) -> None:
        pass
    
    def fit(self, X, Y):
        pass
    
    def predict(self, x) -> np.ndarray:
        pass
    

class LinearRegression(model):
    def __init__(self):
        self.V_m = None
        self.V_n = None
        self.m = None
        self.n = None
        
        return None
    
    def weighted_average(self, Z, Zerr):
        return (Z/Zerr**2).sum()/(1/Zerr**2).sum()
    
    def fit(self, X, Y, Yerr):
        self.y_wa = self.weighted_average(Y, Yerr)
        self.xy_wa = self.weighted_average(X*Y, Yerr)
        self.x_wa = self.weighted_average(X, Yerr)
        self.x_sq_wa = self.weighted_average(X**2, Yerr)
        
        self.sigma_wa = Yerr.size/(1/Yerr**2).sum()
        self.V_m = self.sigma_wa/(X.size*(self.x_sq_wa-self.x_wa**2))
        self.V_n = self.sigma_wa/(X.size*(self.x_sq_wa-self.x_wa**2))*self.x_sq_wa
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
    

def CSV_to_PDF(csv_file_name, output_file_name):
    df = pd.read_csv(csv_file_name)
    html_table = df.to_html()

    options = {    'page-size': 'Letter', 'margin-top': '0mm', 'margin-right': '0mm', 'margin-bottom': '0mm', 'margin-left': '0mm'}

    pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    pdfkit.from_string(html_table, output_file_name, options=options)
    
    


def setSpace(axis, allX, allY, border=0.1, resolution=0.1):
    xRange = np.round(np.max(allX)-np.min(allX), 2)
    yRange = np.round(np.max(allY)-np.min(allY), 2)
    
    xstep = xRange * resolution 
    ystep = yRange * resolution 
    
    xLimits = [np.min(allX)-xRange*border, np.max(allX)+xRange*border+xstep]
    yLimits = [np.min(allY)-yRange*border, np.max(allY)+yRange*border+ystep]
    
    axis.set_xticks(np.arange(xLimits[0], xLimits[1], xstep))
    axis.set_xticks(np.arange(xLimits[0], xLimits[1], xstep * 0.2), minor=True)
    axis.set_yticks(np.arange(yLimits[0], yLimits[1], ystep))
    axis.set_yticks(np.arange(yLimits[0], yLimits[1], ystep * 0.2), minor=True)
    
    axis.set_xlim(xLimits)
    axis.set_ylim(yLimits)
    
    axis.grid(which="both")
    axis.grid(which="minor", alpha=0.4)
    axis.grid(which="major", alpha=0.7)
    
    return axis
    
def plotLine(axis, x, y, label=None, linewidth=0.7, linestyle="solid"):
    axis.plot(x, y, linewidth=0.7, label=label, linestyle=linestyle)
    
    return axis

def plotData(axis, X, Xerr, Y, Yerr, label=None, capsize=3, elinewidth=0.7, fmt=",", polyfit=1):
    axis.errorbar(X, Y, Yerr, Xerr, label=label, capsize=capsize, elinewidth=elinewidth, fmt=fmt)
    modle = None
    
    if polyfit != 0:
        xrange = np.max(X)-np.min(X)
        yrange = np.max(Y)-np.min(Y)
        x = np.linspace(np.min(X)-xrange*0.1, np.max(X)+xrange*0.1, 100)
        
        if polyfit==1:
            model = LinearRegression().fit(X, Y, Yerr)
            y_most_and_least = model.predict_most_and_least(x)
            plotLine(axis, x, y_most_and_least[0], linestyle="dashdot")
            plotLine(axis, x, y_most_and_least[1], linestyle="dashdot")
            y = model.predict(x)
        elif polyift>1:
            y = np.poly1d(np.polyfit(X, Y, polyfit))(x)
            
        
        plotLine(axis, x, y, label=label)
        
    return axis, model