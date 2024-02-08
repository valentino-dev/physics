import numpy as np

dT = 1
dCk = 0.0042
Te = np.array([34.9, 39.8, 39.3])
Tk = np.array([20.4, 20.2, 20.2])
Ck = 0.619
T1 = 99.6
m = np.array([0.155, 0.505, 0.446])
dt = 0.5
C = Ck*(Te-Tk)/(T1-Te)
dC = ((dT*Ck*(Te-Tk)/(T1-Te)**2)**2+(dT*Ck*(T1-Tk)/(T1-Te)**2)**2+(-Ck*dT/(T1-Te))**2+(dCk*(T1-Tk)/(T1-Te))**2)**(1/2)
print("dC", dC)
cm = C/m
dcm = dC/m
lc = np.array([0.897, 0.382, 0.477])
mms = (0.18*0.066+0.08*0.059+0.74*0.056)
print(mms)
mm = np.array([0.027, 0.064, mms])
print((cm/lc-1)*100)
print(dC)
print(C)
print(cm, dcm)
c = cm*mm
print(cm*mm)
dolong = 3*8.3*0.001 
print(dolong)
print(dcm*mm)
print(((c/dolong)-1)*100)


tTr = np.array([17.6, 16.56, 16.45, 16.49])
tTe = np.array([21.32, 20.8, 21.01, 22.31])
tAl = np.array([23.49, 23.76, 23.98, 23.78])
atTr = tTr.sum()/4
atTe = tTe.sum()/4
atAl = tAl.sum()/4

print(atTr)
print(atTe)
print(atAl)
TTr = atTr/50
TTe = atTe/50
TAl = atAl/50
dT = 0.5/4/50
print(TTr)
print(TTe)
print(TAl)
T = np.array([TTr, TTe, TAl])
print(0.5/4/50)
print(T)
m = np.array([4.5, 7.1, 9.4])*0.001
dm = 0.0001
r = 5.95*0.001
dr = 0.00005
dpL = 100
pL = 100000
g = 9.81
V=1.141*10**-3
dp0 = (dpL**2+(dm*g/3.141/r**2)**2+(2*dr*m*g/3.141/r**3)**2)**(1/2)
p0 = pL+m*g/3.141/r**2

k = 4*m*V/T**2/r**4/p0
dk = k*((dm/m)**2+(2*dT/T)**2+(4*dr/r)**2+(dp0/p0)**2)**(1/2)
print(k, dk)
print(p0, dp0)
print(k.sum()/3)
print((dk**2).sum()**(1/2)/3)