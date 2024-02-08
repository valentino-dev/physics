import numpy as np
import math
import matplotlib.pyplot as plt
import straight_fit as sf

h = np.array([100, 75, 50, 25]) / 100
dh = np.ndarray((4,4))+0.005
r025m25t = [10.9, 10.8, 11, 9.2, 7.5, 5.2]
r025m25tn = [9.2, 10.1, 10.1 , 10.3, 10.8, 11.3]
r025m25n = [10.0, 11.0, 11.0, 10.0, 8.0, 6.0]
r10m25t = [2.8, 2.7, 2.8, 2.4, 1.9, 1.3]
r10m25tn = [9.9, 10.1, 10.2, 9.4, 10.3, 10.9]
r10m25n = [11.0, 11.0, 11.0, 9.0, 8.0, 6.0]
r025m50t = [7.5, 7.5, 7.7, 6.4, 5.4, 3.7]
r025m50tn = [10, 10, 9.9, 10.1, 9.1, 11.1]
r025m50n = [16.0, 16.0, 16.0, 15.0, 11.0, 8.0]
r10m50t = [1.9, 2, 2, 1.7, 1.3, 0.9]
r10m50tn = [10.4, 10.3, 9.9, 10.3, 9.9, 10.3]
r10m50n = [15.0, 16.0, 15.0, 14.0, 11.0, 8.0]

t = np.array([r025m25t, r10m25t, r025m50t, r10m50t])
tn = np.array([r025m25tn, r10m25tn, r025m50tn, r10m50tn])
n = np.array([r025m25n, r10m25n, r025m50n, r10m50n])

T = tn/n

aT = np.ndarray((4,4), dtype="float")
for i in range(4):
    aT[i] = np.array([(([T[i][0] + T[i][1] + T[i][2]])[0]/3), T[i][3], T[i][4], T[i][5]])
at = np.ndarray((4,4), dtype="float")
for i in range(4):
    at[i] = np.array([(([t[i][0] + t[i][1] + t[i][2]])[0]/3), t[i][3], t[i][4], t[i][5]])


dt = 0.5
omega = np.ndarray((4,4), dtype="float")
dOmega = np.ndarray((4,4), dtype="float")
dOmega2 = np.ndarray((4,4), dtype="float")
dT = np.ndarray((4,6), dtype="float")
aDT = np.ndarray((4,4), dtype="float")

for i in range(4):
    omega[i] = 2*np.pi/aT[i]
    dT[i] = n[i]**-1*dt
    aDT[i] = np.array([((dT[i][0]/3)**2+(dT[i][1]/3)**2+(dT[i][2]/3)**2)**(1/2), dT[i][3], dT[i][4], dT[i][5]])
    dOmega[i] = 2*np.pi*aT[i]**(-2)*aDT[i]
    dOmega2[i] = 8*np.pi**2*aT[i]**(-3)*aDT[i]
    
y = np.ndarray((4,4), dtype="float")
yPErr = np.ndarray((4,4), dtype="float")
yNErr = np.ndarray((4,4), dtype="float")
omega2 = omega**2

Fits = np.ndarray((4,4), dtype="float")
x = np.repeat([h], 4, 0)
xerr = (np.zeros(4)+0.5)/100
for i in range(4):
    #Fits[i] = Fit.calc_m_n_Sm_Sn(h, xerr, omega2[i], dOmega2[i])
    print("\n\n")
    #y[i] = np.array(x[i]*Fits[i][0]+Fits[i][1])
    #yPErr[i] = np.array(x[i]*(Fits[i][0]+Fits[i][2])+(Fits[i][1]+Fits[i][3]))
    #yNErr[i] = np.array(x[i]*(Fits[i][0]-Fits[i][2])+(Fits[i][1]-Fits[i][3]))

xerr = (np.zeros(4)+0.5)
#print("at", at, "\n\nomega", omega, "\n\ndOmega", dOmega)
Fits = np.ndarray(4, dtype="object")
data = np.ndarray((4, 8), dtype="float")
for i in range(4):
    Fits[i] = sf.Fit(at[i], xerr, omega[i], dOmega[i])
    data[i] = Fits[i].GetFit()
    print("\n\n")
    y[i] = np.array(at[i]*data[i][0]+data[i][1])
    yPErr[i] = np.array(at[i]*(data[i][0]+data[i][2])+(data[i][1]+data[i][3]))
    yNErr[i] = np.array(at[i]*(data[i][0]-data[i][2])+(data[i][1]-data[i][3]))

colors = ["r", "g", "b", "y"]
for i in range(0,4):
#    print(i)
    plt.errorbar(x=at[i], y=omega[i], yerr=dOmega[i], fmt=colors[i]+"+")
    plt.plot(at[i], y[i], colors[i])
    #plt.plot(x[i], yPErr[i], colors[i]+"--", x[i], yNErr[i], colors[i]+"--")

print("omega2\n", np.round(omega2, decimals=1))
print("delta omega2\n", np.round(dOmega2, decimals=2))
print("omeag", omega)
print("at", at)
print("domega", dOmega)
m=np.array([0.025, 0.05])
g=9.81
r=np.array([0.025, 0.1])
print(Fits)
Trägheitsmoment = np.zeros(4)
dI = np.zeros(4)


#Trägheitsmoment[0] = m[0]*g*2/Fits[0][0]-m[0]*r[0]**2 
#Trägheitsmoment[1] = m[0]*g*2/Fits[1][0]-m[0]*r[1]**2 
#Trägheitsmoment[2] = m[1]*g*2/Fits[2][0]-m[1]*r[0]**2 
#Trägheitsmoment[3] = m[1]*g*2/Fits[3][0]-m[1]*r[1]**2
#Trägheitsmoment[0] = m[0]*r[0]*g/Fits[0][0]-m[0]*r[0]**2 
#Trägheitsmoment[1] = m[1]*r[0]*g/Fits[1][0]-m[1]*r[0]**2 
#Trägheitsmoment[2] = m[0]*r[1]*g/Fits[2][0]-m[0]*r[1]**2 
#Trägheitsmoment[3] = m[1]*r[1]*g/Fits[3][0]-m[1]*r[1]**2 
#dI[0] = m[0]*g*2/Fits[0][2]-m[0]*r[0]**2
#dI[1] = m[0]*g*2/Fits[1][2]-m[0]*r[1]**2 
#dI[2] = m[1]*g*2/Fits[2][2]-m[1]*r[0]**2 
#dI[3] = m[1]*g*2/Fits[3][2]-m[1]*r[1]**2
#dI[0] = m[0]*g*2/Fits[0][0]**2*Fits[0][2]
#dI[1] = m[0]*g*2/Fits[1][0]**2*Fits[1][2] 
#dI[2] = m[1]*g*2/Fits[2][0]**2*Fits[2][2] 
#dI[3] = m[1]*g*2/Fits[3][0]**2*Fits[3][2]
#dI[0] = m[0]*r[0]*g/Fits[0][0]**2*Fits[0][2]
#dI[1] = m[0]*r[1]*g/Fits[1][0]**2*Fits[1][2] 
#dI[2] = m[1]*r[0]*g/Fits[2][0]**2*Fits[2][2] 
#dI[3] = m[1]*r[1]*g/Fits[3][0]**2*Fits[3][2]

print("Trägheitsmoment: ", Trägheitsmoment) 
print("DTrägheitsmoment: ", dI) 
plt.show()



    