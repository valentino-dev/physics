
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import olib
import seaborn as sns

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

exec="a"

def a():
    x_tilde_strich = np.array([760, 720, 680, 640, 600, 560, 520, 480, 440, 400])*1e-3
    x_tilde = np.array([157, 156, 155, 154, 151, 149, 147, 146, 137, 128])*1e-3
    B = np.array([187, 169, 150, 133, 114, 97, 80, 65, 47, 31])*1e-3
    G = 22*1e-3
    d = 1*1e-3
    X = 300*1e-3
    x_G = 30*1e-3
    f1=50*1e-3
    f2=-50*1e-3

    dG=0.1*1e-2
    dB=0.2*1e-2
    dX=0.1*1e-2

    x_strich = X-x_tilde_strich
    dx_strich = np.zeros_like(x_strich)+(dX**2+dX**2)**(1/2)
    x_plane = X-x_tilde
    dx_plane = np.zeros_like(x_plane)+(dX**2+dX**2)**(1/2)
    gamma = B/G
    dgamma = ((dB/G)**2+(B/G**2*dG)**2)**(1/2)

    Y = x_plane
    Yerr = dx_plane
    X = 1+1/gamma
    Xerr = 1/gamma**2*dgamma
    print(X, Y)

    X = X*1e2
    Xerr = Xerr*1e2
    Y = Y*1e3
    Yerr = Yerr*1e3

    fig, ax = plt.subplots()
    ax, model = olib.plotData(ax, X, Xerr, Y, Yerr, label="Messwerte", xscaleing=1, yscaleing=1, fmt="v")
    ax = olib.setSpace(ax, X, Y, title="Abbe'sche Verfahren", xlabel=r"$1+\frac{1}{\gamma}\cdot 10^2$", ylabel=r"$x\cdot 10^3[m]$")
    #model.printParameter()

    ax.legend()
    plt.savefig("362a_x.pdf", dpi=500)
    print(olib.roundToError(X, Xerr), olib.roundToError(Y, Yerr))


    X = 1+gamma
    Xerr = dgamma
    Y = x_strich
    Yerr = dx_strich

    X = X*1e1
    Xerr = Xerr*1e1
    Y = Y*1e2
    Yerr = Yerr*1e2

    fig, ax = plt.subplots()
    ax, model = olib.plotData(ax, X, Xerr, Y, Yerr, label="Messwerte", xscaleing=1, yscaleing=1, fmt="v")
    ax = olib.setSpace(ax, X, Y, title="Abbe'sche Verfahren", xlabel=r"$1+\gamma\cdot 10^1$", ylabel=r"$x'\cdot 10^2 [m]$")
    #odel.printParameter()

    ax.legend()
    plt.savefig("362a_x_tilde.pdf", dpi=500)
    #plt.show()
    print(olib.roundToError(X, Xerr), olib.roundToError(Y, Yerr))

def b():
    Untergrundhelligkeit = 0.8
    f_helligkeit = np.array([1204, 1206, 1210, 1152, 795, 1105, 1367, 1470, 1315, 1245, 1235, 1355, 1500, 1607, 1447, 1000, 1050, 1398, 1399, 1392])*1e-2-Untergrundhelligkeit
    g_helligkeit = np.array([805, 905, 995, 920, 700, 530, 735, 1002, 807, 684, 950, 1032, 1036, 942, 734, 1092, 1162, 1172, 1030, 820])*1e-2-Untergrundhelligkeit
    e_helligkeit = np.array([18.30,11.22,11.24,10.52,15.10,15.19,12.05,13.05,13.20,22.62,10.44,11.75,12.40,13.81,13.37,10.68,11.30,12.40,13.92,14.02])-Untergrundhelligkeit
    x = np.array([1,3,5,7,9,1,3,5,7,9,4,4,4,4,4,6,6,6,6,6])
    y = np.array([2,2,2,2,2,4,4,4,4,4,1,2,3,4,5,1,2,3,4,5])
    d = np.zeros_like(f_helligkeit)+0.2
    max = np.array([f_helligkeit, g_helligkeit, e_helligkeit]).max()
    #data_f = pd.dataframe(f_helligkeit, [x, y])
    data_f = np.zeros((9, 5))
    data_f[x-1, y-1] = f_helligkeit
    data_g = np.zeros((9, 5))
    data_g[x-1, y-1] = g_helligkeit
    data_e = np.zeros((9, 5))
    data_e[x-1, y-1] = e_helligkeit
    
    fig, ax = plt.subplots()
    ax, model = olib.plotData(ax, x+y*9, np.zeros_like(f_helligkeit), e_helligkeit, d, label="e", polyfit=0, color="r")
    ax, model = olib.plotData(ax, x+y*9, np.zeros_like(f_helligkeit), f_helligkeit, d, label="f", polyfit=0, color="g")
    ax, model = olib.plotData(ax, x+y*9, np.zeros_like(f_helligkeit), g_helligkeit, d, label="g", polyfit=0, color="b")
    ax.legend()
    ax.set_title("362: Helligkeit")
    ax.set_ylabel("Helligkeit in Lux")
    ax.set_xlabel("Position")
    ax.set_xticks(np.arange(0, (x+y*9).max()+1, 2))
    plt.savefig("362mmPlot.pdf", dpi=500)
    
    fig, ax = plt.subplots()
    ax = sns.heatmap(data_e, annot=True)
    ax.set_title("Heatmap zu 362.e")
    plt.savefig("362e.pdf", dpi=500)
    
    fig, ax = plt.subplots()
    ax = sns.heatmap(data_f, annot=True)
    ax.set_title("Heatmap zu 362.f")
    plt.savefig("362f.pdf", dpi=500)
    
    fig, ax = plt.subplots()
    ax = sns.heatmap(data_g, annot=True)
    ax.set_title("Heatmap zu 362.g")
    plt.savefig("362g.pdf", dpi=500)
    print(olib.roundToError(e_helligkeit, d))
    print(olib.roundToError(f_helligkeit, d))
    print(olib.roundToError(g_helligkeit, d))
    print(x, y)
    
def c():
    f1 = 5*1e-2
    f2 = -5*1e-2
    d = 5*1e-2
    f_theo = (1/f1+1/f2-d/f1/f2)**(-1)
    f1 = 0.049
    df1 = 0.0025
    f2 = .05075
    df2 = 0.0002
    h_plane = 0.089
    dh_plane = 0.0033
    d
    #d = h_plane + h_strich
    #f_hauptebenden = (1/f1+1/f2-d/f1/f2)**(-1)
    #df_hauptebenden = ((f_hauptebenden**-2*df1*(-1/f1**2+d/f1**2/f2))**2+(f_hauptebenden**-2*df2*(-1/f2**2+d/f1/f2**2))**2+(f_hauptebenden**-2*dd))**(1/2)
    #print(f_hauptebenden, df_hauptebenden)
    f_m = (f1+f2)/2
    df_m = ((df1/2)**2+(df2/2)**2)**(1/2)
    print(f_theo)
    print(f_m, df_m)
    print(f_m/f_theo-1)
    
    
    
    
for char in exec:
    locals()[char]()

