import numpy as np
import matplotlib.pyplot as plt
import straight_fit
import math


path="C:\\Users\\angel\\Meine Ablage\\Studium\\Studium Physik\\Praktikum 1\\104va.txt"
f = open(path, "r")
content = f.read()
f.close()

#print(content)
size = 8
rows = 2
arr1 = [""]*size
arr2 = [""]*size

k = 0
h = 0
lastd = "\n"
for i in range(len(content)):
    if(content[i]==" "):
        lastd = " "
        k+=1
    elif(content[i]=="\n"):
        lastd = "\n"
        k+=1
    else:
        if(k%rows==0):
            arr1[int(k/rows)]+=content[i]
        elif(k%rows==1):
            arr2[int(k/rows)]+=content[i]

Zeit = np.array(arr1, dtype="f")
Abstand = np.array(arr2, dtype="f") * 0.01
Periodendauer = Zeit/10
dZeit = np.zeros(Zeit.size) + 0.05
dAbstand = 0.001
M = 0.599



aT2 = Abstand*Periodendauer**2
a2 = Abstand**2
daT2 = ((2*Abstand*Periodendauer*dZeit)**2+(Periodendauer**2*dAbstand)**2)**(1/2)
da2 = 2*Abstand*dAbstand
print("T: ", Periodendauer, "a:", Abstand,  "aT**2:", aT2, "daT**2:", daT2, "a2:", a2, "da2:", da2)

Fit = straight_fit.fit()

m, n, V_m, V_n = Fit.calc_m_n_Vm_Vn(a2, daT2, aT2, daT2)
S_m = (V_m)**(1/2)
S_n = (V_n)**(1/2)
print(m, n)

a = np.array([0, 3, 5.9, 7.3, 8.8, 10.2, 11.7, 13.1, 14.6], dtype="f")*0.01
r = 0.003
R = 0.15
i = a.size
g = 4*math.pi**2/m
dg = 4*math.pi**2/m**2*S_m
I = M*g*n/(4*math.pi**2)
dM=0.001
dI1 = ((g*n/4/math.pi**2*dM)**2+(M*n/4/math.pi**2*dg)**2+(M*g/4/math.pi**2*S_n)**2)**(1/2)
dI2 = (M*g*n/4/math.pi**2)*((dM/M)**2+(dg/g)**2+(S_n/n)**2)**(1/2)
print("g:", g, "+-", dg, "I:", I, "+-", dI1, dI2)
I2 = 1/2*M/(R**2-i*r**2)*(R**4-(r**2*(i*r**2+2*a**2).sum()))
I3 = 1/2*M*R**2
print("I2: ", I2, "I3:", I3)
x = a2 
y = m*x+n
yperr=(m+S_m)*x+n+S_n
ynerr=(m-S_m)*x+n-S_n
#plt.errorbar(a2, aT2, daT2, fmt="b+")
#plt.plot(x, y, "r", x, yperr, "r--", x, ynerr, "r--")


#plt.show()
