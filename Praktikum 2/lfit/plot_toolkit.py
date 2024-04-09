import numpy as np
import matplotlib.pyplot as plt
from lfit import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def setSpace(allX, allY, axis, border=0.1, resolution=0.1):
    xRange = np.max(allX)-np.min(allX)
    yRange = np.max(allY)-np.min(allY)
    xLimits = [np.min(allX)-xRange*border, np.max(allX)+xRange*border]
    yLimits = [np.min(allY)-yRange*border, np.max(allY)+yRange*border]
    axis.set_xticks(np.arange(xLimits[0], xLimits[1], np.round(xRange, 1) * resolution))
    axis.set_xticks(np.arange(xLimits[0], xLimits[1], np.round(xRange, 1) * resolution * 0.2), minor=True)
    axis.set_yticks(np.arange(yLimits[0], yLimits[1], np.round(yRange, 1) * resolution))
    axis.set_yticks(np.arange(yLimits[0], yLimits[1], np.round(yRange, 1) * resolution * 0.2), minor=True)
    axis.set_xlim(xLimits)
    axis.set_ylim(yLimits)
    axis.grid(which="both")
    axis.grid(which="minor", alpha=0.4)
    axis.grid(which="major", alpha=0.7)
    return axis
    
def plotLine(x, y, axis, lable=None, linewidth=0.7, linestyle="solid"):
    axis.plot(x, y, linewidth=0.7, label=label, linestyle=linestyle)
    return axis

def plotData(X, Xerr, Y, Yerr, axis, label=None, capsize=3, elinewidth=0.7, fmt=",", polyfit=1):
    axis.errorbar(X, Y, Yerr, Xerr, label=label, capsize=capsize, elinewidth=elinewidth, fmt=fmt)
    modle = None
    
    if polyfit != 0:
        xrange = np.max(X)-np.min(X)
        yrange = np.max(Y)-np.min(Y)
        x = np.linspace(np.min(x)-xrange*0.1, np.max(x)+xrange*0.1, 100)
        
        if polyfit==1:
            model = LinearRegression().fit(X, Y, Yerr)
            y_most_and_least = model.predict_most_and_least(x)
            plotLine(x, y_most_and_least[0], linestyle="dashdotted")
            plotLine(x, y_most_and_least[1], linestyle="dashdotted")
            y = model.predict(x)
        elif polyift>1:
            y = np.poly1d(np.polyfit(X, Y, polyfit))(x)
            
        
        plotLine(x, y, axis, label=label)
        
    return axis, model
    
    