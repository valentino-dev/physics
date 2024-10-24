from olib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif"})
exec = "c"




def b():
    data = pd.read_csv("Fadenstrahlrohr.csv", sep=",")
    
    U = data["U"].to_numpy()
    I0 = data["I0"].to_numpy()
    I180 = data["I180"].to_numpy()
    I = np.concatenate([I0, I180])
    d1 = np.tile((178 - data["d1"].to_numpy())*1e-3, 2)
    dd1 = 2e-3
    mu0 = 4*3.14159*1e-7
    n = 130
    R = 0.150


    
    r = d1/2
    dr = dd1/2
    #fdsfdsfsfds Fehler
      = (4/5)**(3/2)*mu0*n*(I0-I180)/R
    
    B_rounded, err = roundToError(B*1e7, digits=4)
    print(B_rounded)
    outData1 = pd.DataFrame(np.array([B_rounded]).T, columns=[r"B$\cdot 10^7$"])
    outData1.to_csv("242_Tabelle_1.csv")
    
    X = np.tile(U, 2)
    Xerr = np.zeros_like(X)
    Y = (r*I)**2*1e5
    Yerr = 2*r*I*dd1*1e5
    print(X.shape, Xerr.shape, Y.shape, Yerr.shape)
    fig, ax = plt.subplots()
    print(X, Y)
    ax, model = plotData(ax, X, Xerr, Y, Yerr, polyfit=1, fmt="x", label="Daten")
    model.printParameter()
    spezifische_ladung = 2*R**2/(model.m*(4/5)**3*mu0**2*n**2)
    d_spezifische_ladung = 2*R**2/(model.m*(4/5)**3*mu0**2*n**2)**2*model.m*(4/5)**3*mu0**2*n**2*model.V_m**(1/2)
    print(roundToError(np.array([spezifische_ladung]), xerr=np.array([d_spezifische_ladung])))
    abweichung = spezifische_ladung/1.75882e11-1
    print(abweichung)
    Be = 1/2*(4/5)**(3/2)*mu0*n*(I180-I0)/R
    print(Be.mean())
    print(Be.mean()/20e-6-1)
    
    ax = setSpace(ax, X, Y, title="Abb. 1: Bestimmung der spezifischen Ladung", xlabel=r"U[V]", ylabel=r"$r^2B^2\cdot 10^5$[$m^2T^2$]")
    plt.legend()
    plt.savefig("Bestimmung der spezifischen Ladung", dpi=500)
    X, Xerr = roundToError(X, xerr=Xerr)
    Y, Yerr = roundToError(Y, xerr=Yerr)
    outData = pd.DataFrame(np.array([X, Xerr, Y, Yerr]).T, columns=[r"U[V]", r"$\Delta$U[V]", r"$r^2B^2\cdot 10^5$[$m^2T^2$]", r"$\Delta r^2B^2\cdot 10^5$[$m^2T^2$]"])
    outData.to_csv("242_Tabelle_2.csv")


def c():
    #data = np.array([pd.read_csv("Drop01.csv").to_numpy(),
            #pd.read_csv("Drop02.csv", sep=",").to_numpy(),
            #pd.read_csv("Drop03.csv").to_numpy(),
            #pd.read_csv("Drop04.csv").to_numpy(),
            #pd.read_csv("Drop05.csv").to_numpy(),
            #pd.read_csv("Drop06.csv").to_numpy(),
            #pd.read_csv("Drop07.csv").to_numpy(),
            #pd.read_csv("Drop08.csv").to_numpy(),
            #pd.read_csv("Drop09.csv").to_numpy(),
            #pd.read_csv("Drop10.csv").to_numpy(),
            #pd.read_csv("Drop11.csv").to_numpy(),
            #])
    
    
    data = np.loadtxt("Messung242_f_3_columns.txt")
    data[:, [1, 0]] = data[:, [0, 1]]
    data = data.reshape(12, 3, 3)
    print(data)
    dropcount = 12
    messungperdrop = 3
    
    dt = 0.3
    dd = np.zeros_like(data)+0.025e-3
    #dd = 0
    d = np.zeros_like(data)+5e-4

    velocity = np.divide(d,data,out=np.zeros_like(d), where=data!=0)
    dVelocity = ((np.divide(dd,data,out=np.zeros_like(dd), where=data!=0)**2+(np.divide(d,data,out=np.zeros_like(d), where=data!=0))**2*dt)**2)**(1/2)
    predVel0 = velocity[:,:,2]-velocity[:,:,1]
    dPredVel0 = (dVelocity[:,:,2]**2+dVelocity[:,:,1]**2)**(1/2)
    print(velocity, dVelocity, predVel0, dPredVel0)
    
    out_data = []
    for i in range(dropcount):
        out_data.append(pd.DataFrame({"predVal0": predVel0[i,:], "dPredVel0": dPredVel0[i,:], "v0*2": velocity[i,:,0]*2, "d(v0*2)": dVelocity[i,:,0]*2}))
        out_data[i].to_csv(f"242_Drop{i}_Abschätzung.csv")
    
    
    #abweichung
    Valid = [True]*dropcount
    for i in range(dropcount):
        print(f"Tröpfchen {i+1}") 
        for k in range(messungperdrop):
            
            test1 = (predVel0[i,k]+dPredVel0[i, k])>velocity[i,k,0]*2-dVelocity[i, k,0]*2
            test2 = (predVel0[i, k]-dPredVel0[i,k])<(velocity[i,k,0]*2)+dVelocity[i, k,0]*2
            #print(test1, test2)
            if test1 and test2 and predVel0!=0 and dPredVel0!=0:
                pass
            else:
                Valid[i] = False
        print(f"Valid => {Valid[i]}")
        #abweichung = np.round(((velocity[i,:,2]-velocity[i,:,1])/(velocity[i,:,0]*2)-1)*100, 2)
        #print(abweichung)
        
    for i in range(dropcount):
        X = np.tile(np.arange(messungperdrop),2)
        Y = np.concatenate([predVel0[i], velocity[i,:,0]*2])
        Yerr = np.concatenate([dPredVel0[i], dVelocity[i,:,0]*2])
        #print(X.shape, Y.shape, Yerr.shape)
        #plt.errorbar(X, Y, Yerr, fmt="x", capsize=4)
        #plt.show()       #ax[0, i].legend()

# counts, bins = np.histogram(data.flatten(), 20)
# ax.stairs(counts, bins)
#for i in range(data.shape[0]):
#    if i == data.shape[0]-1:
#        ax[0, 1].plot(data[i], np.arange(data.shape[1]), label=f"Markov Iter. {i}", linewidth=0.5)
#        counts, bins = np.histogram(data[i], 20)
#        ax[1, 1].stairs(counts, bins)
        
#ax[0, 0].set_xlim((-4, 4))
#ax[0, 1].set_xlim((-4, 4))
    
#ax[0, 0].legend()
#ax[0, 1].legend()
    
    ethaeff = 18.19e-6
    g=9.81
    rhoöl = 886
    rholuft = 1.2041
    E = 500/7.67e-3
    COEFF = 9*ethaeff/(4*g*(rhoöl-rholuft))
    r = (np.abs(COEFF*(velocity[:,:,2]-velocity[:,:,1])))**(1/2)
    dr = ((r**(-1)*COEFF*dVelocity[:,:,2])**2+(r**(-1)*COEFF*dVelocity[:,:,1])**2)**(1/2)
    
    print(f"r: {r.mean(axis=1)[Valid]}+-{dr.mean(axis=1)[Valid]}")
    
    COEFF2 = 3*3.14159*ethaeff*r/E
    q_notsummed = (COEFF2*(velocity[:,:,2]+velocity[:,:,1]))
    q = np.mean(q_notsummed, axis=1)
    dq_notsumed = ((COEFF2*dVelocity[:,:,2])**2 + (COEFF2*dVelocity[:,:,1])**2 + (COEFF2/r*dr*(velocity[:,:,2]+velocity[:,:,1]))**2)**(1/2)
    dq = np.sum((dq_notsumed/dq_notsumed.shape[0])**2, axis=1)**(1/2)
    print(f"q: {q[Valid]}+-{dq[Valid]}")
    X = np.arange(len(Valid))[Valid]+1
    X = X*10
    print(Valid, q)
    Y = q[Valid]*1e20
    Yerr = dq[Valid]*1e20
    fig, ax = plt.subplots()
    ax, model = plotData(ax, X, np.zeros_like(X), Y, Yerr, polyfit=0, fmt="x", label="Tröpfchen")
    
    X2 = np.zeros(7)
    Y2 = np.zeros(7)+1.6e-19*np.arange(len(X2))*1e20
    for i in range(Y2.size):
        ax = plotLine(ax, np.array([0, 200]), np.tile(Y2[i], 2), color="g")
    
    print(X, Y)
    ax = setSpace(ax, X, Y, xlabel=r"Tröpfchen Nummer $\cdot 10$", ylabel=r"$Ne\cdot 10^{20}$", title=r"Abb. 2: Bestimmung der Elementarladung")
    Y, Yerr = roundToError(Y, xerr=Yerr)
    outData = pd.DataFrame(np.array([X, Y, Yerr]).T, columns=[r"Tröpfchen Nummer $\cdot 10$", r"$\Delta Ne\cdot 10^{20}$", r"$Ne\cdot 10^{20}$"])
    outData.to_csv("242_Tabelle_3.csv")
    rr = r.mean(axis=1)*1e9
    drr = np.linalg.norm(dr/r.shape[0], axis=1)*1e9
    rr, drr = roundToError(rr, drr)
    outData2 = pd.DataFrame(np.array([rr, drr]).T, columns=[r"$r$[m]$\cdot 10^9$",r"$\delta r$[m]$\cdot 10^9$"])
    outData2.to_csv("242_Tabelle_4.csv")
    plt.legend()
    plt.savefig("Bestimmung der Elementarladung", dpi=500)
    
    Teiler = []
    for i in range(len(Y)):
        Teiler.append(Y[i]/np.arange(10))
        print(f"{i}: {Teiler[i]}")
    res = {"T2": [4, 18.75], "T7": [3, 23.3], "T8": [3, 20.76], "T9": [2, 20.6], "T10": [3, 18.9]}
    print(res)
    #res = {"T2": [5, 15.00], "T7": [4, 17.5], "T8": [4, 15.575], "T9": [2, 20.6], "T10": [4, 14.175]}
    rr = rr*1e-9
    drr = drr*1e-9
    X = 1/rr[Valid]*1e-4
    Xerr = X**2*drr[Valid]
    
    de = dq[Valid]/np.array([4, 3, 3, 2, 3])
    e = q[Valid]/np.array([4, 3, 3, 2, 3])
    print(f"dq: {dq[Valid]}")
    Yerr = 2/3*(e)**((2/3)-1)*de*1e15
    #Yerr = (q[Valid]/np.array([5, 4, 4, 2, 4]))**(2/3)
    #Yerr = (q[Valid]/np.array([4, 3, 3, 2, 3]))**(2/3)
    Y = (np.array([18.75, 23.33, 20.76, 20.6, 18.9])*1e-20)**(2/3)*1e15
    #Y = (np.array([15.00, 17.5, 15.575, 20.6, 14.175])*1e-20)**(2/3)
    fig, ax = plt.subplots()
    ax, model = plotData(ax, X, np.zeros_like(X), Y, Yerr, fmt="x")
    model.printParameter()
    print(model.n**(3/2))
    print(X, Y)
    ax = setSpace(ax, X, Y, title="Abb. 3: Bestimmung der korregierten Elementarladung", xlabel=r"$r\cdot 10^{-4}$", ylabel=r"$q \cdot 10^{14}$ [C]")
    X, Xerr = roundToError(X, digits=4)
    Y, Yerr = roundToError(Y, Yerr)
    outDate = pd.DataFrame(np.array([X, Y, Yerr]).T, columns=[r"$r\cdot 10^{-4}$", r"$q \cdot 10^{14}$ [C]", r"$\delat q \cdot 10^{14}$ [C]"])
    outDate.to_csv("242_Tabelle_5.csv")
    plt.savefig("Bestimmung der korregierten Elementarladung", dpi=500)
    em = 1.751405 * 1e11
    dem = 1.6*1e5
    eee = 1.851*1e-19
    deee = 0.066*1e-19
    m = eee/em
    dm = ((deee/em)**2+(eee/em**2*dem)**2)**(1/2)
    print(m, dm)
    #plt.show()
    
    
    
        
        
    
        
    
    
    


def g():
    pass


def h():
    pass


def i():
    pass



for char in exec:
    locals()[char]()
