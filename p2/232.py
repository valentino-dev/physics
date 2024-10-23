import numpy as np
import straight_fit as sf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt 
import matplotlib
font = {"fontname": "Computer Modern", "family": "serif"}
#plt.rcParams['text.usetex'] = True

capsize = 3
elinewidth = 0.5
def setSpace(allX, allY):
    border = 0.1
    resolution = 0.1
    xRange = np.max(allX)-np.min(allX)
    yRange = np.max(allY)-np.min(allY)
    xLimits = [np.min(allX)-xRange*border, np.max(allX)+xRange*border]
    yLimits = [np.min(allY)-yRange*border, np.max(allY)+yRange*border]
    plt.xticks(np.arange(xLimits[0], xLimits[1], np.round(xRange, 1) *resolution))
    plt.xticks(np.arange(xLimits[0], xLimits[1], np.round(xRange, 1)*resolution*0.2), minor=True)
    plt.yticks(np.arange(yLimits[0], yLimits[1], np.round(yRange, 1)*resolution))
    plt.yticks(np.arange(yLimits[0], yLimits[1], np.round(yRange, 1)*resolution*0.2), minor=True)
    plt.xlim(xLimits[0], xLimits[1])
    plt.ylim(yLimits[0], yLimits[1])
    plt.grid(which="both")
    plt.grid(which="minor", alpha=0.4)
    plt.grid(which="major", alpha=0.7)
    
# 232.a
def Part_a():
    
    U0 = np.array([1.871,2.215,2.634,3.128,3.582,3.934,4.453]) # V
    U = np.array([18.1,21.5,25.5,30.2,35.0,38.2,43.5])*(5/50) # V
    dU = np.zeros(len(U))+0.2*(5/50) # V
    I = np.array([4,5,6,7,8,9,10])*(500/50)*1e-3 # A
    dI = np.zeros(len(I))+0.2*(500/50)*1e-3 # A
    
    model = LinearRegression().fit(I.reshape(-1, 1), U.reshape(-1, 1), 1/dU**2)
    m, n = model.predict(np.array([0, 1]).reshape(-1, 1))
    print(f"{m}+{(n-m)}x")
    plt.title("232.a und b: U-I-Abhängigkeit", font)
    plt.xlabel("I[µA]", font)
    plt.ylabel("U[V]", font)
    plt.errorbar(I*1e3, U , dU, dI*1e3, fmt=",", color="#D97345", label="Datenpunkte", capsize=capsize, elinewidth=elinewidth)
    plt.plot(I*1e3, model.predict(I.reshape(-1, 1)), color="#D97345", label="R_A Gerade")
    plt.plot(I*1e3, 59.3*I, color="#579991", label="R_x Gerade")
    plt.legend()
    setSpace(I*1e3, U)
    #plt.show()
    plt.savefig("232.a_U-I-Abhängigkeit.png", dpi=500)
    
#Part_a()
# 232.f
def Part_f():
    R_y = np.array([200, 400, 600, 800, 1000])/1000*100
    y = np.array([200, 400, 600, 800, 1000])
    dy = np.zeros_like(y) + 2
    dR = 0.2
    l = np.zeros(y.size) + 1000
    
    dU = 0.2/50*5
    U_inf = np.array([9.0, 12.9, 21.5, 35.5, 44.5])*5/50
    U_20 = np.array([6.0, 11.0, 16.0, 27.5, 41.0])*5/50
    U_50 = np.array([7.0, 12.5, 18.8, 22.2, 42.5])*5/50
    dU = np.zeros(U_inf.size) + 0.02
    
    
    model_inf = LinearRegression().fit((y/l).reshape(-1, 1), U_inf.reshape(-1, 1), 1/dU**2)
    model_20 = LinearRegression().fit((y/l).reshape(-1, 1), U_20.reshape(-1, 1), 1/dU**2)
    model_50 = LinearRegression().fit((y/l).reshape(-1, 1), U_50.reshape(-1, 1), 1/dU**2)
    #m, n = model.predict(np.array([0, 1]).reshape(-1, 1))
    #print(f"{m}+{(n-m)}x")
    
    plt.title("232.f: U-R-Abhängigkeit", font)
    plt.xlabel("x/l", font)
    plt.ylabel("U[V]", font)
    
    #plt.errorbar(U_inf, y, dy)
    plt.errorbar(y/l, U_inf , dU, dy/l, fmt=",", color="#D9BB45", label="Datenpunkte für R = inf", capsize=capsize, elinewidth=elinewidth)
    plt.plot(y/l, model_inf.predict((y/l).reshape(-1, 1)), color="#D9BB45", label="Gerade für R = inf", linewidth=0.7)
    
    plt.errorbar(y/l, U_20 , dU, dy/l, fmt=",", color="#D945D0", label="Datenpunkte für R = 20 Ohm", capsize=capsize, elinewidth=elinewidth)
    plt.plot(y/l, model_20.predict((y/l).reshape(-1, 1)), color="#D945D0", label="Gerade für R = 20", linewidth=0.7)
    
    plt.errorbar(y/l, U_50 , dU, dy/l, fmt=",", color="#45D9C8", label="Datenpunkte für R = 50 Ohm", capsize=capsize, elinewidth=elinewidth)
    plt.plot(y/l, model_50.predict((y/l).reshape(-1, 1)), color="#45D9C8", label="Gerade für R = 50", linewidth=0.7)
    
    setSpace(y/l, [U_inf, U_20, U_50])
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(1)
    #plt.show()
    plt.savefig("232.f_U-R-Abhängigkeit.png", dpi=500)
#Part_f()


def Part_g():
    R_y = np.array([200, 400, 600, 800, 1000])/1000*100
    y = np.array([200, 400, 600, 800, 1000])
    dy = np.zeros_like(y) + 2
    dR = 0.2
    l = np.zeros(y.size) + 1000
    
    dU = 0.2/50*5
    U_inf = np.array([9.0, 12.9, 21.5, 35.5, 44.5])*5/50
    U_20 = np.array([6.0, 11.0, 16.0, 27.5, 41.0])*5/50
    U_50 = np.array([7.0, 12.5, 18.8, 22.2, 42.5])*5/50
    #U_50 = np.array([7.0, 12.5, 18.8, 30.2, 42.5])*5/50
    dU = np.zeros(U_inf.size) + 0.02
    
    P_inf = U_inf**2/R_y
    dP_inf = ((2*U_inf*dU/R_y)**2+(U_inf**2/R_y**2*dR)**2)**(1/2)
    
    P_20 = U_20**2/R_y
    dP_20 = ((2*U_20*dU/R_y)**2+(U_20**2/R_y**2*dR)**2)**(1/2)
    
    P_50 = U_50**2/R_y
    dP_50 = ((2*U_50*dU/R_y)**2+(U_50**2/R_y**2*dR)**2)**(1/2)
    
    
    
    #model_inf = LinearRegression().fit(y.reshape(-1, 1), P_inf.reshape(-1, 1), 1/dP_inf*2)
    #model_20 = LinearRegression().fit(y.reshape(-1, 1), P_20.reshape(-1, 1), 1/dP_20**2)
    #model_50 = LinearRegression().fit(y.reshape(-1, 1), P_50.reshape(-1, 1), 1/dP_50**2)
    
    model_inf = np.poly1d(np.polyfit(y, P_inf, 2))
    model_20 = np.poly1d(np.polyfit(y, P_20, 2))
    model_50 = np.poly1d(np.polyfit(y, P_50, 2))
    
    #m, n = model.predict(np.array([0, 1]).reshape(-1, 1))
    #print(f"{m}+{(n-m)}x")
    
    plt.title("232.g: P-x-Abhängigkeit", font)
    plt.xlabel("x[Skt]", font)
    plt.ylabel("P(x)[W]", font)
    Y = np.linspace(np.min(y), np.max(y), 100)
    #plt.errorbar(U_inf, y, dy)
    plt.errorbar(y, P_inf , dP_inf, dy, fmt=",", color="#D9BB45", label="Datenpunkte für R = inf", capsize=capsize, elinewidth=elinewidth)
    plt.plot(Y, model_inf(Y), color="#D9BB45", label="Gerade für R = inf", linewidth=0.7)
    
    plt.errorbar(y, P_20 , dP_20, dy, fmt=",", color="#D945D0", label="Datenpunkte für R = 20 Ohm", capsize=capsize, elinewidth=elinewidth)
    plt.plot(Y, model_20(Y), color="#D945D0", label="Gerade für R = 20", linewidth=0.7)
    
    plt.errorbar(y, P_50 , dP_50, dy, fmt=",", color="#45D9C8", label="Datenpunkte für R = 50 Ohm", capsize=capsize, elinewidth=elinewidth)
    plt.plot(Y, model_50(Y), color="#45D9C8", label="Gerade für R = 50", linewidth=0.7)
    
    setSpace(y, [P_inf, P_20, P_50])
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(1)
    #plt.show()
    plt.savefig("232.g_P-x-Abhängigkeit.png", dpi=500)
    
#Part_g()
    
# 232.d
def Part_d():
    R = np.linspace(13,143,10)
    U = np.array([12.1,21.9,24.0,25.9,26.9,27.5,28.0,29.0,29.0,29.5])*(5/50) # V
    dU = np.zeros(len(U))+0.2*(5/50) # V
    I = np.array([29.8,13.6,10.9,8.2,7.1,6.1,5.8,4.7,4.1,3.5])*(500e-3/50) # A
    dI = np.zeros(len(I))+0.2*(500e-3/50) # A
    R1 = 20 # Ohm
    Rx = R1
    R2 = 50 # Ohm
    Ry = R2
    U0 = 4 # V
    Ri = 50 # Ohm
    
    model = LinearRegression().fit(I.reshape(-1, 1), U.reshape(-1, 1), 1/dU**2)
    m, n = model.predict(np.array([0, 1]).reshape(-1, 1))
    print(f"{m}+{(n-m)}x")
    
    plt.title("232.d: U-I-Abhängigkeit", font)
    plt.xlabel("I[µA]", font)
    plt.ylabel("U[V]", font)
    plt.errorbar(I*1e3, U , dU, dI*1e3, fmt=",", color="#D97345", label="Datenpunkte", capsize=capsize, elinewidth=elinewidth)
    plt.plot(I*1e3, model.predict(I.reshape(-1, 1)), color="#D97345", label="R_A Gerade")
    
    setSpace(I*1e3, U)
    plt.legend()
    #plt.show()
    plt.savefig("232.d_U-I-Abhängigkeit.png", dpi=500)
#Part_d()    

# 232.m
def Part_m():
    R_1 = np.array([1.1,0.9,0.7,0.65,0.57,0.49,0.43,0.38,0.35,0.30,0.27,0.24,0.21,0.18,0.16,0.14,0.13,0.12,0.095,0.085])
    dR_1 = np.zeros(len(R_1))+0.02
    R_2 = np.array([4.2,4.1,4.1,4.0,4.1,4.2,4.0,4.2,4.1,4.1,4.1,4.2,4.1,4.6,4.4,4.2,4.2,4.1,4.2,4.2])
    dR_2 = np.zeros(len(R_2))+0.1
    R_3 = np.array([112,127,138,170,203,209,330,460,640,1350,4600,15e+3,55e+3,160e3,230e+3,280e+3,300e+3,320e+3,350e+3,360e+3])
    #R_3 = np.array([112,127,138,170,203,209,330,460,640,1350,4600,15e+3,55e+3,8.8e+6,230e+3,280e+3,300e+3,320e+3,350e+3,360e+3])
    dR_3 = np.array([10,10,10,10,10,30,30,30,30,200,200,200,2e+4,2e+4,2e+4,2e+4,2e+4,2e+4,2e+4,2e+4])
    R_4 = np.array([1.1,1.1,1.12,1.14,1.15,1.17,1.18,1.20,1.20,1.21,1.22,1.23,1.25,1.27,1.28,1.30,1.31,1.33,1.35,1.35])
    dR_4 = np.zeros(len(R_4))+0.02
    R_5 = np.array([100,100.2,100,100,100,100,100.1,100,100.2,100,100,100.1,100,100,100.3,100.2,100.1,100.3,100.2,100.2])
    dR_5 = np.zeros(len(R_5))+0.1
    R = np.array([R_1, R_2, R_3, R_4, R_5])
    dR = np.array([dR_1, dR_2, dR_3, dR_4, dR_5])

    T_1 = np.array([19.8, 24.1, 28.2, 32.6, 37.0, 40.7, 44.1, 47.2, 50.4, 53.6, 57.1, 61.4, 64.8, 68.2, 73.6, 78.4, 81.6, 85.4, 90.5, 95.2])
    T_2 = np.array([20.5, 25.1, 28.7, 33.5, 37.5, 41.6, 45.0, 48.2, 50.9, 54.7, 57.8, 61.9, 65.4, 70.3, 74.3, 79.1, 82.3, 87.8, 91.2, 96.1])
    T_3 = np.array([21.2, 25.8, 29.34, 34.1, 38.6, 42.4, 45.4, 48.8, 51.9, 55.0, 58.8, 62.5, 66.1, 70.9, 75.9, 79.7, 82.9, 88.8, 92.4, 97.3])
    T_4 = np.array([22.3, 26.5, 30.7, 34.9, 39.4, 43.1, 46.3, 49.4, 52.6, 56.1, 59.9, 63.6, 67.1, 72.0, 76.9, 80.4, 83.6, 89.5, 93.4, 98.4])
    T_5 = np.array([23.5, 26.9, 31.5, 35.7, 39.9, 44.1, 47.2, 50.2, 53.4, 57.0, 60.5, 63.9, 67.7, 73.0, 77.8, 81.3, 84.9, 90.0, 94.2, 98.6])
    T = np.array([T_1, T_2, T_3, T_4, T_5]) 
    dT = np.zeros((5, 20)) + 0.2
    
    X_1 = 1/(T_1+273.15)*1e3
    dX_1 = dT[0]/(X_1+273.15)**2*1e3
    X_2 = T_2
    dX_2 = dT[1]
    X_3 = T_3 
    dX_3 = dT[2]
    X_4 = T_4
    dX_4 = dT[3]
    X_5 = T_5
    dX_5 = dT[4]
    X = np.array([X_1, X_2, X_3, X_4, X_5])
    dX = np.array([dX_1, dX_2, dX_3, dX_4, dX_5])
    
    Y_1 = np.log(R_1)
    dY_1 = dR[0]/R[0]
    Y_2 = R_2
    dY_2 = dR_2
    Y_3 = np.log(R_3)
    dY_3 = dR_3/R_3
    Y_4 = R_4
    dY_4 = dR_4
    Y_5 = R_5
    dY_5 = dR_5
    Y = np.array([Y_1, Y_2, Y_3, Y_4, Y_5])
    dY = np.array([dY_1, dY_2, dY_3, dY_4, dY_5])
    
    #model_1 = PolynomialFeatures(2)
    #Y = np.linspace(np.min(T_1), np.max(T_5), 100)
    models_sklearn = []
    models_own = []
    x = []
    for i in [0, 1, 2, 3, 4]:
        if i == 2:
            models_sklearn.append(0)
            models_own.append(0)
            x.append(0)
            continue
        models_sklearn.append(LinearRegression().fit(X[i].reshape(-1, 1), Y[i].reshape(-1, 1), sample_weight=1/dY[i]**2))
        models_own.append(sf.Fit(X[i], dX[i], Y[i], dY[i]))
        models_own[i].GetFit()
        x.append(np.linspace(np.min(X[i]), np.max(X[i]), 100))
    
    

    plot = 0
    if plot == 0:
         
        plt.xlabel("T[°C]", font)
        plt.ylabel("R(T)", font)
        for i in [4]: 
            plt.title(f"232.n: Kohleschicht-Widerstand", font)
            plt.errorbar(X[i], Y[i], dY[i], dX[i], fmt=",", label=f"Widerstand {i+1}", capsize=capsize, elinewidth=elinewidth)
            print(models_own[i].m/models_own[i].n, ((models_own[i].V_m**(1/2)/models_own[i].n)**2+(models_own[i].m/models_own[i].n**2*models_own[i].V_n**(1/2))**2)**(1/2), models_own[i].m, models_own[i].n)
            setSpace(X[i], Y[i])
            plt.plot(x[i], models_own[i].predict(x[i]))
            plt.plot(x[i], models_sklearn[i].predict(x[i].reshape(-1, 1)))
            
    if plot == 1:
        plt.title("232.n: Platin-Widerstand", font)
        plt.xlabel("1000/T[1/K]", font)
        plt.ylabel("log(R(T))", font)
        for i in [0]:
            plt.errorbar(X[i], Y[i], dY[i], dX[i], fmt=",", label=f"Widerstand {i+1}", capsize=capsize, elinewidth=elinewidth)
            setSpace(X[i], Y[i])
            plt.plot(x[i], models_own[i].predict(x[i]))
            k=1.38064852e-23
            print(models_own[i].m*2*k, models_own[i].V_m**(1/2)*2*k, models_own[i].m, models_own[i].n)
            
    if plot == 2:
        plt.title("232.n: PTC-Widerstand", font)
        plt.xlabel("T[°C]", font)
        plt.ylabel("log(R(T))", font)
        for i in [2]:
            plt.errorbar(X[i], Y[i], dY[i], dX[i], fmt="x", label=f"Widerstand {i+1}", capsize=capsize, elinewidth=elinewidth)
            setSpace(X[i], Y[i])
    if plot == 3:  
        for i in range(T.shape[0]):
            plt.errorbar(T[i], R[i]/np.max(R[i]), dR[i]/np.max(R[i]), dT[i], fmt="x", label=f"Widerstand {i+1}", capsize=capsize, elinewidth=elinewidth)
    
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(1)
    plt.savefig("232.n_Widerstand_5.png", dpi=500)
    #plt.show()

Part_m()
    
    
