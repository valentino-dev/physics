from olib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif"})
exec = "bc"



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
    B  = (4/5)**(3/2)*mu0*n*(I0-I180)/R
    
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
    data = np.array([pd.read_csv("Drop01.csv").to_numpy(),
            pd.read_csv("Drop02.csv").to_numpy(),
            pd.read_csv("Drop03.csv").to_numpy(),
            pd.read_csv("Drop04.csv").to_numpy(),
            pd.read_csv("Drop05.csv").to_numpy(),
            pd.read_csv("Drop06.csv").to_numpy(),
            pd.read_csv("Drop07.csv").to_numpy(),
            pd.read_csv("Drop08.csv").to_numpy(),
            pd.read_csv("Drop09.csv").to_numpy(),
            pd.read_csv("Drop10.csv").to_numpy(),
            pd.read_csv("Drop11.csv").to_numpy(),
            ])
    
    dt = 0.3
    dd = 0.025e-3
    #dd = 0
    d = 5e-4
    velocity = d/data
    dVelocity = ((dd/data)**2+(d/data**2*dt)**2)**(1/2)
    predVel0 = velocity[:,:,2]-velocity[:,:,1]
    dPredVel0 = (dVelocity[:,:,2]**2+dVelocity[:,:,1]**2)**(1/2)
    out_data = []
    for i in range(11):
        out_data.append(pd.DataFrame({"predVal0": predVel0[i,:], "dPredVel0": dPredVel0[i,:], "v0*2": velocity[i,:,0]*2, "d(v0*2)": dVelocity[i,:,0]*2}))
        out_data[i].to_csv(f"242_Drop{i}_Abschätzung.csv")
    
    
    #abweichung
    Valid = [True]*11
    for i in range(11):
        print(f"Tröpfchen {i+1}") 
        for k in range(5):
            test1 = (predVel0[i,k]+dPredVel0[i, k])>velocity[i,k,0]*2-dVelocity[i, k,0]*2
            test2 = (predVel0[i, k]-dPredVel0[i,k])<(velocity[i,k,0]*2)+dVelocity[i, k,0]*2
            #print(test1, test2)
            if test1 and test2:
                pass
            else:
                Valid[i] = False
        print(f"Valid => {Valid[i]}")
        #abweichung = np.round(((velocity[i,:,2]-velocity[i,:,1])/(velocity[i,:,0]*2)-1)*100, 2)
        #print(abweichung)
        
    for i in range(11):
        X = np.tile(np.arange(5),2)
        Y = np.concatenate([predVel0[i], velocity[i,:,0]*2])
        Yerr = np.concatenate([dPredVel0[i], dVelocity[i,:,0]*2])
        #print(X.shape, Y.shape, Yerr.shape)
        #plt.errorbar(X, Y, Yerr, fmt="x", capsize=4)
        #plt.show()
    
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
    r = r.mean(axis=1)*1e9
    dr = np.linalg.norm(dr/r.shape[0], axis=1)*1e9
    r, dr = roundToError(r, dr)
    outData2 = pd.DataFrame(np.array([r, dr]).T, columns=[r"$r$[m]$\cdot 10^9$",r"$\delta r$[m]$\cdot 10^9$"])
    outData2.to_csv("242_Tabelle_4.csv")
    plt.legend()
    plt.savefig("Bestimmung der Elementarladung", dpi=500)
    
    
    


def g():
    pass


def h():
    pass


def i():
    pass



for char in exec:
    locals()[char]()
