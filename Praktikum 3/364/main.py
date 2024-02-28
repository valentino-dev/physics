import olib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

exec = "g"

def a():
    B_10x = 10e-3
    B_20x = 10e-3
    G_10x = 9e-6*1e2
    G_20x = 5.3e-6*1e2
    dG = 0.1e-6*1e2
    
    B_10x_10x = 15.5e-2
    B_20x_5x = 6.8e-2
    G_10x_10x = 10e-6*1e2
    G_20x_5x = 5e-6*1e2
    dB = 0.2e-2
    
    s0=25e-2
    b=35e-2
    db = 0.5e-2
    
    gamma_obj_10x_10x = B_10x/G_10x
    gamma_M_10x_10x = B_10x_10x/G_10x_10x
    gamma_oku_10x_10x = gamma_M_10x_10x/gamma_obj_10x_10x
    
    dgamma_obj_10x_10x = B_10x/G_10x**2*dG
    dgamma_M_10x_10x = B_10x_10x/G_10x_10x**2*dG
    dgamma_oku_10x_10x = ((dgamma_M_10x_10x/gamma_obj_10x_10x)**2+(gamma_M_10x_10x/gamma_obj_10x_10x**2*dgamma_obj_10x_10x))**(1/2)
    
    
    gamma_obj_20x_5x = B_20x/G_20x
    gamma_M_20x_5x = B_20x_5x/G_20x_5x
    gamma_oku_20x_5x = gamma_M_20x_5x/gamma_obj_20x_5x
    
    dgamma_obj_20x_5x = dB/G_20x
    dgamma_M_20x_5x = dB/G_20x_5x
    dgamma_oku_20x_5x = ((dgamma_M_20x_5x/gamma_obj_20x_5x)**2+(gamma_M_20x_5x/gamma_obj_20x_5x**2*dgamma_obj_20x_5x)**2)**(1/2)
    
    V1010 = gamma_M_10x_10x/gamma_obj_10x_10x*s0/b
    V2005 = gamma_M_20x_5x/gamma_obj_20x_5x*s0/b
    
    dV1010 = V1010*((dgamma_M_10x_10x/gamma_M_10x_10x)**2+(dgamma_obj_10x_10x/gamma_obj_10x_10x)**2+(db/b)**2)**(1/2)
    dV2005 = V2005*((dgamma_M_20x_5x/gamma_M_20x_5x)**2+(dgamma_obj_20x_5x/gamma_obj_20x_5x)**2+(db/b)**2)**(1/2)
    
    print(f"obj1010: {gamma_obj_10x_10x} +- {dgamma_obj_10x_10x}; M_1010: {gamma_M_10x_10x} +- {dgamma_M_10x_10x}; oku_1010: {gamma_oku_10x_10x} +- {dgamma_oku_10x_10x}\nobj_2005: {gamma_obj_20x_5x} +- {dgamma_obj_20x_5x}; M_2005: {gamma_M_20x_5x} +- {dgamma_M_20x_5x}; oku_2005: {gamma_oku_20x_5x} +- {dgamma_oku_20x_5x}")
    print(f"V1010: {V1010} +- {dV1010}; V2005: {V2005} +- {dV2005}")
    

def b():
    B1 = 8.9e-2
    B2 = 12.9e-2
    dB = 0.1e-2
    
    G1 = 200e-6
    G2 = 200e-6
    
    T = 6e-2
    dT = 0.1e-2
    
    gamma1 = B1/G1
    dgamm1 = dB/G1
    
    gamma2 = B2/G2
    dgamm2 = dB/G2
    
    f = T/(gamma2-gamma1)
    df = ((dT/(gamma1-gamma2))**2+(T/(gamma1-gamma2)**2*dgamm1)**2+(T/(gamma1-gamma2)**2*dgamm2)**2)**(1/2)
    print(f"gamma1: {gamma1} +- {dgamm1}; gamma2: {gamma2} +- {dgamm2}; f: {f} +- {df}")
    
def calc(x, xerr):
    R = 17/2*1e-2
    E=6.63
    dE=0.02
    
    
    r = R-x
    dr = xerr
    
    d = r*2*3.141/36
    dd = dr*2*3.141/36
    
    #d=r
    #dd=dr
    
    alpha=d/E
    dalpha=((dd/E)**2+(d/E**2*dE)**2)**(1/2)
    return alpha, dalpha
    
    
    
    
def g():
    
    # Latex font
    plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif"})
    
    data = np.array([0.6, 1.0, 2.0, 3.0])*1e-3
    data_err = np.zeros(4)
    
    datay = np.array([4.7, 5.5, 7.0, 7.6])*1e-2
    datay_err = (np.zeros(4)+0.2)*1e-2
    
    
    
    # Defining X and Y
    X, Xerr = (1/data), (data_err/data**2)
    X, Xerr = X*1e-1, Xerr*1e-1
    Y, Yerr = calc(datay, datay_err)
    Y, Yerr = Y*1e5, Yerr*1e5
    
    print(X, Xerr)
    print(Y, Yerr)

    # Defining Subplots for olib
    fig, ax = plt.subplots()

    # Printing (with fmt=","), Fitting (with polyfit=1; no fit with polyfit=0 and fit on n polynomial with polyfit=n) and Predicting (Automaticly with fitting) Data
    ax, model = olib.plotData(ax, X, Xerr, Y, Yerr, label="Messwerte", polyfit=1, fmt="v")

    # Defining print Space with grid lines, labels and etc.
    ax = olib.setSpace(ax, X, Y, title="Auflösungsvermögen", xlabel=r"$\frac{1}{D}\cdot10^{-1}$", ylabel=r"$\alpha\cdot 10^4$")

    # Print the parameters of the fittet model
    model.printParameter()

    # Show plot
    #plt.show()
    ax.legend()

    # Print Plot
    plt.savefig("346g.pdf", dpi=500)

for char in exec:
    locals()[char]()