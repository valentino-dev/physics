import numpy as np
import matplotlib.pyplot as plt
import olib
import scipy.odr as odr

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
exec = "b"

def a():
    
    U0 = np.array([.1,.2,.5,.6,.6,.8,.8,.7,.7,.8,.8,.7,.7,.8,.9,.8,.7,.8])+2
    U0 = U0*1e-3
    U0err = np.zeros_like(U0)+0.1
    U0err = U0err*1e-3
    U0_mean_err = np.linalg.norm(U0err)/13
    U0_mean = U0[5:].mean()
    X = np.arange(10, 190, 10)
    Xerr = np.zeros_like(X)+0.5
    table_U0 = olib.Table(X, Xerr, U0, U0err, "372.a: Offsetspannung", r"$t$[s]", r"$U$[V]")
    fig, ax = plt.subplots()
    ax = olib.plotData(ax, table_U0, polyfit=0)
    ax = olib.setSpace(ax, table_U0)
    table_U0.saveAsPDF()
    plt.savefig("372.a Offsetspannung.pdf", dpi=500)
    print(U0_mean, U0_mean_err)
    
    
    t5 = np.arange(10,310,10)

    U5 = np.array([90.0,129.0,140.0,146.0,149.0,151.0,154.0,155.9,157.4,159.0,160.2,161.3,162.3,163.1,163.8,164.6,165.2,165.7,166.2,166.8,167.3,167.7,168.1,168.5,168.9,169.4,169.6,170.0,170.3,170.5])*1e-3
    U5err = np.zeros_like(U5)+0.1*1e-3
    U5err[0:3] = 5*1e-3
    U5err[3:8] = 1*1e-3
    Y = U5/U5.max()
    Yerr = ((U5err/U5.max())**2+(U5/U5.max()**2*0.1*1e-3)**2)**(1/2)
    table = olib.Table(t5, np.zeros_like(t5)+0.5, Y, Yerr, "372.a: Ansprechzeit", r"$t[\textrm{s}]$", r"$\frac{U}{U_{\textrm{max}}}$")
    fig, ax = plt.subplots()
    ax = olib.plotData(ax, table, polyfit=0)
    ax.plot(np.arange(0, 300), np.zeros(300)+90)
    ax = olib.setSpace(ax, table)
    table.saveAsPDF()
    plt.savefig("372.a Ansprechzeit.pdf", dpi=500)
    
def b():
    V = 100
    S = 29*1e-6# µV m² 1/W
    A = 10.4*9.9*1e-4
    DA = ((0.1*9.9)**2+(0.1*10.4)*2)**0.5*1e-4
    
    U0 = 2.769*1e-3
    dU0 = 0.033*1e-3
    
    T = np.array([29,33,38,44,48,53,59,64,69,75])+273.15
    dT = np.zeros_like(T)+1
    T0 = 25+273.15
    dT0 = 0.1
    T4 = T**4-T0**4

    U_schwarz = np.array([15.0,19.4,25.9,31.1,38.0,44.4,51.6,58.6,66.3,73.4])*1e-3-U0
    DU_schwarz = np.zeros_like(U_schwarz)+0.1*1e-3
    U_matt = np.array([4.7,7.0,7.0,8.0,9.6,10.7,12.9,13.8,15.9,16.9])*1e-3-U0
    DU_matt = np.array([1,1,1,.1,.1,.1,.1,.1,.1,.1])*1e-3
    U_weiß = np.array([17.5,21.0,26.0,31.9,38.5,44.9,51.9,59.4,66.4,74.5])*1e-3-U0
    DU_weiß = np.array([1,1,1,.1,.1,.1,.1,.1,.1,.1])*1e-3
    U_glänzend = np.array([4.5,6.0,4.7,5.5,6.5,6.6,8.2,8.7,10.5,11.1])*1e-3-U0
    DU_glänzend = np.array([2,2,2,.1,.1,.1,.1,.1,.1,.1])*1e-3
    
    I_schwarz = U_schwarz/V/S
    dI_schwarz = ((DU_schwarz)**2+(dU0)**2)**(1/2)/V/S
    I_matt = U_matt/V/S
    dI_matt = ((DU_matt)**2+(dU0)**2)**(1/2)/V/S
    I_weiß = U_weiß/V/S
    dI_weiß = ((DU_weiß)**2+(dU0)**2)**(1/2)/V/S
    I_glänzend = U_glänzend/V/S
    dI_glänzend = ((DU_glänzend)**2+(dU0)**2)**(1/2)/V/S
    
    X = T4
    Xerr = ((4*T**3*dT)**2+(4*T0**3*dT0)**2)**(1/2)
    Y = T4*5.67e-8
    #table_theo.x_scaling, table_theo.y_scaling = table_weiß.getScaling()
    
    title = r"372.b: Stefan-Boltzman-Gesetz"
    xlable = r"$T^4$[$\textrm{K}^4$]"
    ylable = r"$\frac{\Phi}{A}$[$\frac{\textrm{W}}{\textrm{m}^2}$]"
    
    table_theo = olib.Table(X, Xerr, Y, np.zeros_like(Y), title, xlable, ylable)
    
    table_weiß = olib.Table(X, Xerr, I_weiß, dI_weiß, "Weiße Lackierung", xlable, ylable)
    table_weiß.x_scaling, table_weiß.y_scaling = table_theo.getScaling()
    table_schwarz = olib.Table(X, Xerr, I_schwarz, dI_schwarz, "Schwarze Lackierung", xlable, ylable)
    table_schwarz.x_scaling, table_schwarz.y_scaling = table_theo.getScaling()
    table_matt = olib.Table(X, Xerr, I_matt, dI_matt, "Mattes Metall", xlable, ylable)
    table_matt.x_scaling, table_matt.y_scaling = table_theo.getScaling()
    table_glänzend = olib.Table(X, Xerr, I_glänzend, dI_glänzend, "Poliertes Metall", xlable, ylable)
    table_glänzend.x_scaling, table_glänzend.y_scaling = table_theo.getScaling()
    
    fig, ax = plt.subplots()
    print(X, Y)
    print(table_weiß.Y)
    ax, model_schwarz = olib.plotData(ax, table_schwarz, "schwarze Lackierung", polyfit=1, color="#1b9e77")
    ax, model_weiß = olib.plotData(ax, table_weiß, "weiße Lackierung", polyfit=1, color="#d95f02")
    ax, model_matt = olib.plotData(ax, table_matt, "mattes Metall", polyfit=1, color="#7570b3")
    ax, model_glänzend = olib.plotData(ax, table_glänzend, "poliertes Metall", polyfit=1, color="#e7298a")
    model_schwarz.printParameter()
    print(olib.roundToError(np.array([model_schwarz.m/5.67e-8]), np.array([model_schwarz.V_m**(1/2)/5.67e-8])))
    model_weiß.printParameter()
    print(olib.roundToError(np.array([model_weiß.m/5.67e-8]), np.array([model_weiß.V_m**(1/2)/5.67e-8])))
    model_matt.printParameter()
    print(olib.roundToError(np.array([model_matt.m/5.67e-8]), np.array([model_matt.V_m**(1/2)/5.67e-8])))
    model_glänzend.printParameter()
    print(olib.roundToError(np.array([model_glänzend.m/5.67e-8]), np.array([model_glänzend.V_m**(1/2)/5.67e-8])))
    # table_weiß.saveAsPDF(height=-1)
    # table_schwarz.saveAsPDF(height=-1)
    # table_matt.saveAsPDF(height=-1)
    #table_glänzend.saveAsPDF(height=-1)
    
    ax = olib.plotLine(ax, table_theo.X*table_theo.x_scaling, table_theo.Y*table_theo.y_scaling, label="schwarzer Körper")
    
    
    table_joint = olib.Table(X, Xerr, np.append(np.append(np.append(I_schwarz, I_matt), table_theo.Y), I_glänzend), np.append(np.append(np.append(dI_schwarz, dI_matt), dI_weiß), dI_glänzend), title, xlable, ylable)
    print(table_joint.y_scaling)
    ax = olib.setSpace(ax, table_joint)
    ax.legend()
    plt.savefig("372.b_w_theo.pdf", dpi=500)
    
def c():
    pass
    
    
    
for char in exec:
    locals()[char]()
    
    
    
    