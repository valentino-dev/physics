from olib import *
import numpy as n
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif"})
execute = "h"

def c():
    print("Executing c")
    data = pd.read_csv("236/c_data.csv", sep=",")
    R1 = 100
    R2 = 5
    U0 = 2.2
    X = data["R"]#*1e-2
    Xerr = np.zeros_like(data["R"].to_numpy()) 
    Y = 1/data["phi"]#*1e3
    Yerr = 1/data["phi"]**2*data["dphi"]#*1e3
    fig, ax = plt.subplots()
    #ax = setSpace(ax, X, Y, title="Abb. 1: 236.c", xlabel=r"$R\cdot10^{-2}$[$\Omega$]", ylabel=r"$\frac{1}{\varphi}\cdot 10^3$")
    ax, model = plotData(ax, X, Xerr, Y, Yerr, fmt="x", label="Messwerte")
    
    cI = (R1+R2)/((model.m)*U0*R2)
    dcI = (R1+R2)/((model.m)**2*U0*R2)*(model.V_m)**(1/2)
    print(cI,"+-",dcI)
    
    Rg = model.n/(R1+R2)*cI*U0*R2
    dRg = 11
    dRg = ((model.V_n**(1/2)/(R1+R2)*cI*U0*R2)**2+(model.n/(R1+R2)*U0*R2*dcI)**2)**(1/2)
    print(Rg, "+-", dRg)
    
    model.printParameter()
    #plt.savefig("236/Abb_1_236_c", dpi=500)
    
def g():
    print("Executing g")
    data = pd.read_csv("236/g_data.csv", sep=",")
    m = data["m"].to_numpy()
    phi = data["phi"].to_numpy()
    dphi = data["dphi"].to_numpy()
    U = 2.2
    Ra = 1000
    Rg = 149.4311653
    dRg = 11
    #Rg = 
    c = phi/(U/(Ra+Rg))
    dc = ((dphi/(U/(Ra+Rg)))**2+(phi/(U/(Ra+Rg))**2*(U/(Ra+Rg)**2)*dRg)**2)**(1/2)
    print(roundToError(c, dc))
    #rounded_c, rounded_c_err= roundedToErr(c, dc)
    
    #cI = 404763.8097052169
    #dcI = 10449.412886742779
    

def h():
    data = pd.read_csv("236/h_data.csv", sep=",")
    t = data["t"].to_numpy()
    dt = np.zeros_like(t)+0.3
    phi = data["phi"].to_numpy()
    dphi = data["dphi"].to_numpy()
    
    X = t
    Xerr = dt
    Y = np.log(phi)
    Yerr = dphi/phi
    
    fig, ax = plt.subplots()
    #ax = setSpace(ax, X, Y, title="Abb. 2: 236.h", xlabel=r"$t\cdot10^1$[s]", ylabel=r"$\log{\varphi}\cdot10^2$")
    ax, model = plotData(ax, X, Xerr, Y, Yerr, label="Messdaten", fmt="x")
    C = 10e-6
    print(model.printParameter())
    print((model.m/C, model.V_m**(1/2)/C))
    #plt.savefig("236/Abb_2_236_h", dpi=500)
    
    
    
for char in execute:
    locals()[char]()