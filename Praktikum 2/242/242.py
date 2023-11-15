from olib import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({"text.usetex": True,
                     #"font.family": "serif"})
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
    B  = (4/5)**(3/2)*mu0*n*(I0-I180)/R
    
    B_rounded, err = roundToError(B)
    print(B_rounded)
    
    X = np.tile(U, 2)
    Xerr = np.zeros_like(X)
    Y = (r*I)**2
    Yerr = 2*r*I*dd1
    print(X.shape, Xerr.shape, Y.shape, Yerr.shape)
    fig, ax = plt.subplots()
    print(X, Y)
    ax, model = plotData(ax, X, Xerr, Y, Yerr, polyfit=1, fmt="x")
    model.printParameter()
    spezifische_ladung = 2*R**2/(model.m*(4/5)**3*mu0**2*n**2)
    d_spezifische_ladung = 2*R**2/(model.m*(4/5)**3*mu0**2*n**2)**2*model.m*(4/5)**3*mu0**2*n**2*model.V_m**(1/2)
    print(roundToError(np.array([spezifische_ladung]), xerr=np.array([d_spezifische_ladung])))
    abweichung = spezifische_ladung/1.75882e11-1
    print(abweichung)
    Be = 1/2*(4/5)**(3/2)*mu0*n*(I180-I0)/R
    print(Be.mean())
    print(Be.mean()/20e-6-1)
    plt.show()
    
    outData = pd.DataFrame({"B": B_rounded})
    outData.to_csv("242_Tabelle.csv")


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
    for i in range(11):
        print(f"Tröpfchen {i+1}") 
        valid = True
        for k in range(5):
            test1 = (predVel0[i,k]+dPredVel0[i, k])>velocity[i,k,0]*2-dVelocity[i, k,0]*2
            test2 = (predVel0[i, k]-dPredVel0[i,k])<(velocity[i,k,0]*2)+dVelocity[i, k,0]*2
            #print(test1, test2)
            if test1 and test2:
                pass
            else:
                valid = False
        print(f"Valid => {valid}")
        #abweichung = np.round(((velocity[i,:,2]-velocity[i,:,1])/(velocity[i,:,0]*2)-1)*100, 2)
        #print(abweichung)
        
    for i in range(11):
        X = np.tile(np.arange(5),2)
        Y = np.concatenate([predVel0[i], velocity[i,:,0]*2])
        Yerr = np.concatenate([dPredVel0[i], dVelocity[i,:,0]*2])
        print(X.shape, Y.shape, Yerr.shape)
        #plt.errorbar(X, Y, Yerr, fmt="x", capsize=4)
        #plt.show()


def g():
    pass


def h():
    pass


def i():
    pass



for char in exec:
    locals()[char]()