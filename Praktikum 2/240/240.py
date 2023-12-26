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
B = data[2]*1e-3

n_mu_A_dot = 52
n_mu_max_dot = 101

x_scale = 1e-1
y_scale = 1e2

N = 500
l = 477*1e-3
d = 2*1e-3
mu0 = sc.mu_0

H = N*I/l-d/mu0/l*B
H = -H

X = H*x_scale
Y = B*y_scale


#plt.scatter(X, Y, marker=".", linewidths=0.)


fig, ax = plt.subplots()

ax = olib.setSpace(ax, X, Y, title=r"Abb. 1: gesammte Hysteresekurve", xlabel=r"$H\cdot 10^{-1}$[$\frac{A}{m}$]", ylabel=r"$B\cdot 10^{2}$[T]")
ax.plot(X, Y, linewidth=1)


plt.savefig("Abb_1_Hysteresekurve.png", dpi=500)
#plt.show()

fig, ax = plt.subplots()

mask = H>0
X = H*x_scale
Y = B*y_scale
mu_A_a = B[n_mu_A_dot]/H[n_mu_A_dot]
mu_A = mu_A_a/sc.mu_0
mu_max_a = B[n_mu_max_dot]/H[n_mu_max_dot]
mu_max = mu_max_a/sc.mu_0
print(f"mu_A: {mu_A}, mu_max: {mu_max}")
X2 = np.linspace(0, H.max())*x_scale

ax = olib.setSpace(ax, H[mask]*x_scale, B[mask]*y_scale, title=r"Abb. 2: kleiner Ausschnitt der Hysteresekurve", xlabel=r"$H\cdot 10^{-1}$[$\frac{A}{m}$]", ylabel=r"$B\cdot 10^{2}$[T]")
#ax.scatter(X, Y, linewidths=0.5, marker=".")
ax.plot(X, Y, linewidth=1)
ax.plot(X2, X2*mu_max_a*y_scale*1e1, color="g", label=r"$\mu_{max}$", linewidth=0.7)
ax.plot(X2, X2*mu_A_a*y_scale*1e1, color="r", label=r"$\mu_A$", linewidth=0.7)
ax.legend()




plt.savefig("Abb_2_Hysteresekurve.png", dpi=500)
#plt.show()


fig, ax = plt.subplots()
print(H)
mask = H>0
mask = mask*(H<2.5e3)
#print(np.array(-H).max())
X = H*x_scale
Y = B*y_scale
ax = olib.setSpace(ax, H[mask]*x_scale, B[mask]*y_scale, title=r"Abb. 3: großer Ausschnitt der Hysteresekurve", xlabel=r"$H\cdot 10^{-1}$[$\frac{A}{m}$]", ylabel=r"$B\cdot 10^{2}$[T]")
ax.scatter(X, Y, linewidths=0.5, marker="x")
ax.plot(X2, X2*mu_max_a*y_scale*1e1, color="g", label=r"$\mu_{max}$", linewidth=0.7)
ax.plot(X2, X2*mu_A_a*y_scale*1e1, color="r", label=r"$\mu_A$", linewidth=0.7)
ax.legend()



plt.savefig("Abb_3_Hysteresekurve.png", dpi=500)
