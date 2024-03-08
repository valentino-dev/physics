import numpy as np
import matplotlib.pyplot as plt
import olib

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

exec = "a"

def a():
    x_data = 179-np.array([131.5+12/60, 130.5+11/60, 130.5+9/60, 130.0+20/60, 129.5+24/60, 129+27/60, 129+13/60, 128+28/60, 127.5+2/60])
    xerr_data = np.zeros_like(x_data) + 1/60
    y_data = np.array([634.85, 579.06, 576.96, 546.08, 508.58, 479.95, 467.81, 435.83, 404.66])
    yerr_data = np.zeros_like(y_data)
    
    x = x_data*1e1
    xerr = xerr_data*1e1
    y = y_data
    yerr = yerr_data
    
    
    
    fig, ax = plt.subplots()
    ax, model = olib.plotData(ax, x, xerr, y, yerr, fmt="v", polyfit=0, label="Messwerte")
    
    ax = olib.setSpace(ax, x, y, title="Kalibrationskurve", ylabel=r"$\lambda$", xlabel=r"$\delta\cdot 10^1$")
    ax.legend()
    print(olib.roundToError(x, xerr))
    print(olib.roundToError(y, yerr))
    # plt.show()
    plt.savefig("366c.pdf", dpi=500)
    
def calc(x):
    gamma = 60*3.141/180
    return np.sin((x+gamma)/2)/np.sin(gamma/2)
def calcErr(x, dx):
    gamma = 60
    dgamma = 0.012
    return (((1/2*dx*(np.cos((x+gamma)/2)/np.sin(gamma/2)))**2+(1/2*dgamma*(np.cos((x+gamma)/2)/np.sin(x/2)+np.sin((x+gamma/2))/np.sin(gamma/2)**2*np.cos(gamma/2))))**2)**(1/2)
    
def b():
    x_data = np.array([634.85, 579.06, 576.96, 546.08, 508.58, 479.95, 467.81, 435.83, 404.66])*1e-9
    xerr_data = np.zeros_like(x_data)
    y_data = (179-np.array([131.5+12/60, 130.5+11/60, 130.5+9/60, 130.0+20/60, 129.5+24/60, 129+27/60, 129+13/60, 128+28/60, 127.5+2/60]))*3.141/180
    yerr_data = (np.zeros_like(y_data) + 1/60)*3.141/180
    
    x = 1/x_data**2*1e-11
    xerr = xerr_data*1e-11
    y = calc(y_data)*1e3
    yerr =calcErr(y_data, yerr_data)*1e3
    
    fig, ax = plt.subplots()
    
    ax, model = olib.plotData(ax, x, xerr, y, yerr, label="Messwerte")
    #model.printParameter()
    ax = olib.setSpace(ax, x, y, title="Auflösungsvermögen", xlabel=r"$\frac{1}{\lambda^2}\cdot 10^{-11}$", ylabel=r"$n\cdot 10^{3}$")
    ax.legend()
    
    Lambda = np.array([400, 500, 600])*1e-9
    A = (1/2*model.m/Lambda**3)*32.5e-3
    dA = A/model.m*model.V_m**(1/2)
    dLambda = Lambda/A
    ddLambda = Lambda/A**2*dA
    #print(dLambda, ddLambda)
    #print(A, dA)
    
    #print(olib.roundToError(x, xerr))
    #print(olib.roundToError(y, yerr))
    
    
    
    
    #plt.show()
    plt.savefig("366e.pdf", dpi=500)
    
    
    


for char in exec:
    locals()[char]()