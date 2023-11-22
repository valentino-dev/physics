import numpy as np
import array_to_latex as a2l
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as sc
import olib

plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif"})
data = pd.read_csv("240_b_Messung_1.txt", sep="\t", decimal=",").to_numpy().T

print(data.shape)

I = -data[1]
B = data[2]*1e3

N = 500
l = 477*1e-3
d = 2*1e-3
mu0 = sc.mu_0

H = N*I/l-d/mu0/l*B
H = -H

X = H*1e-5
Y = B*1e-1


#plt.scatter(X, Y, marker=".", linewidths=0.)


fig, ax = plt.subplots()

ax = olib.setSpace(ax, X, Y, title=r"Abb. 1: gesammte Hysteresekurve", xlabel=r"$H\cdot 10^{-5}$[$\frac{A}{m}$]", ylabel=r"$B\cdot 10^{-1}$[T]")
ax.plot(X, Y, linewidth=1)


plt.savefig("Abb_1_Hysteresekurve.png", dpi=500)
#plt.show()

fig, ax = plt.subplots()

mask = H>0
X = H*1e-5
Y = B*1e-1
a1 = 29.894/17.1789
a2 = 29.002/19.5815
X2 = np.linspace(0, X.max())

ax = olib.setSpace(ax, H[mask]*1e-5, B[mask]*1e-1, title=r"Abb. 2: kleiner Ausschnitt der Hysteresekurve", xlabel=r"$H\cdot 10^{-5}$[$\frac{A}{m}$]", ylabel=r"$B\cdot 10^{-1}$[T]")
ax.scatter(X, Y, linewidths=0.5, marker=".")
ax.plot(X2, X2*a1, color="g", label=r"$\mu_{max}$")
ax.plot(X2, X2*a2, color="r", label=r"$\mu_A$")
ax.legend()


plt.savefig("Abb_2_Hysteresekurve.png", dpi=500)
#plt.show()


fig, ax = plt.subplots()

mask = H>0
mask = mask*(H<1e6)
#print(np.array(-H).max())
X = H*1e-4
Y = B*1e-0
print(X)
ax = olib.setSpace(ax, H[mask]*1e-4, B[mask]*1e-0, title=r"Abb. 3: großer Ausschnitt der Hysteresekurve", xlabel=r"$H\cdot 10^{-4}$[$\frac{A}{m}$]", ylabel=r"$B$[T]")
ax.scatter(X, Y, linewidths=0.5, marker=".")
X2 = X2*1e1
ax.plot(X2, X2*a1, color="g", label=r"$\mu_{max}$")
ax.plot(X2, X2*a2, color="r", label=r"$\mu_A$")
ax.legend()
print(a1*1e-4/sc.mu_0, a2*1e-4/sc.mu_0)


plt.savefig("Abb_3_Hysteresekurve.png", dpi=500)
