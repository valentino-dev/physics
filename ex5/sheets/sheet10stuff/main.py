import numpy as np
import matplotlib.pyplot as plt

e_ladung = 1.6e-19

#1
K=0.1535e6#*e_ladung
rho=1.7
ZA=0.49954
I=78*e_ladung
m_e=9.1e-31
c=3e8

z=2#*e_ladung
M=7294*m_e

def W_max(M, betagamma):
    return 2*m_e*c**2*betagamma**2/(1+2*m_e/M*(1+betagamma**2)**(1/2)+(m_e/M)**2)

def BB(z, M, betagamma):
    beta=1/(betagamma**-2+1)**(1/2)
    return K*rho*ZA*z**2/beta**2*(np.log(2*m_e*c**2*betagamma**2*W_max(M, betagamma)/I**2)-2*beta**2)


betagamma = 10**np.linspace(-2, 2, int(1e4))


x = betagamma
y = BB(z, M, betagamma)
plt.plot(x, y)
plt.title("Bethe-Bloch-Formel")
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\beta\gamma$')
plt.ylabel(r'$-\langle\frac{\text{d}E}{\text{d}x}\rangle$')
plt.grid()
plt.savefig('nr1.png', dpi=500)

#2
M=np.array([7294, 1836, 206])*m_e
z=np.array([2, 1, 1])
kinetic_energy = W_max(M, 3)
print("p und ekin", (2*M*kinetic_energy)**(1/2), kinetic_energy)


#3

def vel(m0, E_kin):
    #return (E_kin/m0*(1+E_kin/m0/c**2))**(1/2)
    #return ((E_kin+m0*c**2)**2-(m0*c**2)**2)**(1/2)/m0/c
    return c*(1-1/(E_kin/m0/c**2+1)**2)**(1/2)
def betagammafunc(M, T):
    return 1/(c**2/vel(M, T)**2-1)**(1/2)
z=1
M=1836*m_e
R0=1e-6
T_min=0.1*1e6*e_ladung
T0=np.array([10, 100, 1000])*1e6*e_ladung
res = int(1e4)
T = np.linspace(T_min, T0, res)
#print(np.max(T), M, M/2/np.max(T))
#print(vel(M, np.max(T)))
R=R0+np.sum(BB(z, M, betagammafunc(M, T))**-1*T/res/e_ladung/1e2, axis=0)
print("R", R)


#4
T0=100e6*e_ladung
z=1
M=1836*m_e
#dd=1e
d=1e1
res=int(1e3)#int(d/dd)#
def E_kin(d):
    T = np.zeros(res)
    BB_log = np.zeros(res)
    T[0]=T0
    for i in range(1, res):
        #print("T", T[i-1])
        #print("v", vel(M, T[i-1]))
        #print("b", betagammafunc(M, T[i-1]))
        #print("B", BB(z, M, betagammafunc(M, T[i-1]))*e_ladung)
        BB_log[i-1]=BB(z, M, betagammafunc(M, T[i-1]))
        T[i]=T[i-1]-BB_log[i-1]*d/res*e_ladung
    BB_log[-1]=BB(z, M, betagammafunc(M, T[-1]))
        
    return BB_log
x = np.linspace(0, d, res)
y = E_kin(d)

plt.clf()
plt.plot(x, y)

plt.ylabel(r'$\langle\frac{\text{d}E}{\text{d}x}\rangle$/eV')
plt.xlabel(r'Distance $d$/cm')
plt.grid()
plt.title('Energieverlus pro Wegstrecke eines Protons in Graphit')
plt.yscale('linear')
plt.xscale('linear')
plt.savefig('nr4.png', dpi=500)








