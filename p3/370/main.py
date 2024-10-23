import numpy as np
import matplotlib.pyplot as plt
import olib
import scipy.odr as odr

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
# Hier "a", "b" oder "c" angeben, um die jeweilige Funktion auszuführen
exec = "a"

def func(beta, x):
    return beta[0] + beta[1]*np.cos(x*beta[2]+beta[3])**2



def a():
    Um = np.array([8.95,8.26,7.55,6.80,6.02,5.26,4.54,3.88,3.28,2.81,2.41,2.17,2.04,2.05,2.19,2.49,2.90,3.39,4.04,4.70,5.42,6.23,7.04,7.81,8.45,9.10,9.78,10.22,10.53,10.75,10.78,10.70,10.55,10.19,9.77,9.20])
    Up = np.array([8.97,8.34,7.57,6.79,6.07,5.25,4.57,3.88,3.30,2.81,2.44,2.16,2.04,2.05,2.20,2.47,2.87,3.36,3.97,4.67,5.50,6.16,6.88,7.70,8.41,9.02,9.76,10.13,10.45,10.70,10.80,10.74,10.60,10.22,9.83,9.26])
    dU = 0.01
    dphi = 1
    U = np.append(Um, Up)
    print(U.shape)
    phi = np.arange(0, 360, 5)
    print(phi.shape)
    X = phi
    Xerr = np.zeros_like(X)+dphi
    Y = U
    Yerr = np.zeros_like(Y)+dU
    
    
    table = olib.Table(X, Xerr, Y, Yerr, title="Malus Gesetz", xlabel=r"$\phi$[Grad]", ylabel=r"$U$[V]")
    
    fig, ax = plt.subplots()
    ax, model = olib.plotData(ax, table, polyfit=0, label="data")
    ax = olib.setSpace(ax, table)
    model = odr.Model(func)
    fit = odr.ODR(odr.RealData(table.X, table.Y, table.Xerr, table.Yerr), model, [2, 8, 0.01, 1])
    fit.set_job(fit_type=2)
    output = fit.run()
    print(output.beta)
    x = np.linspace(0, 360, 3600)
    y = func(output.beta, x)
    ax = olib.plotLine(ax, x*table.x_scaling, y*table.y_scaling, "prediction")
    I_max = y.max()
    I_min = y.min()
    PG = (I_max-I_min)/(I_max+I_min)
    print("PG: ", PG)
    output.pprint()
    table.saveAsPDF()
    

    
    plt.savefig("370/Malus Gesetz.pdf", dpi=500)
    
    

def b():
    phi = np.array([[22,2,12,8,17], np.array([-16,-42,-41,-12,-10]), np.array([-45,-46,-46,-47,-48]), np.array([-68, -63, -67, -63, -68]), np.array([-84, -86, -85, -86, -84]), np.array([77, 77, 79, 76, 78])-180, np.array([66, 63, 61, 59, 65])-180])
    dphi = np.zeros_like(phi)+10
    phi_bined = phi.mean(1)
    dphi_bined = np.linalg.norm(dphi[:], axis=1)/5
    print(dphi_bined)
    Lambda = np.array([430, 458, 488, 520, 568, 620, 694])*1e-9
    fig, ax = plt.subplots()
    X = 1/Lambda**2
    Xerr = np.zeros_like(Lambda)
    Y = phi_bined
    Yerr = dphi_bined
    table = olib.Table(X, Xerr, Y, Yerr, title="Drehvermoegen", ylabel=r"$\phi$[Grad]", xlabel=r"$\frac{1}{\lambda^2}$[$\frac{1}{m^2}$]")
    ax, model = olib.plotData(ax, table)
    ax = olib.setSpace(ax, table)
    model.printParameter()
    print(model.predict(1/589.3e-9**2)/4)
    table.saveAsPDF(height=-1)
    
    plt.savefig("370/Drehvermögen.pdf", dpi=500)

def c():
    phi = 90-np.array([[36, 39, 36, 36, 35], [56, 58, 55, 57, 60], [73, 68, 69, 66, 70], [76, 76, 74, 75, 75], [89, 91, 89, 88, 88]])
    c = np.flip(np.arange(1, 6))
    print(c)
    print(c.shape)
    dc = np.zeros_like(c)
    dphi = np.zeros_like(phi)+4
    phi_bined = phi.mean(1)
    print(phi_bined)
    
    dphi_bined = np.linalg.norm(dphi, axis=1)/5 
    fig, ax = plt.subplots()
    
    X = c
    Xerr = dc
    Y = phi_bined
    Yerr = dphi_bined
    
    table = olib.Table(X, Xerr, Y, Yerr, title="370.c: Drehvermoegen", xlabel="$c$[mol/L]", ylabel="$\phi$[Grad]")
    
    print(X, Y)
    phi_unk = 90-np.array([78, 74, 74, 74, 76]).mean()
    
    
    ax, model = olib.plotData(ax, table, label="data")
    ax = olib.setSpace(ax, table)
    table.saveAsPDF(height=-1)
    model.printParameter()
    print((phi_unk-model.n)/model.m)
    print(phi_unk)
    
    
    plt.savefig("370/370.c_ Drehvermoegen.pdf", dpi=500)
    
    
    
for char in exec:
    locals()[char]()
    
    
    
    