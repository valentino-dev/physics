import olib
import numpy as np

X = np.array([1, 2, 3, 4])
Xerr = np.zeros_like(X)+0.5
Y = X**2
Yerr = np.zeros_like(Y)+0.5
table = olib.Table(X, Xerr, Y, Yerr, title=r"Sandbox plot $\delta$", xlabel=r"x $\lambda$", ylabel=r"y $\omega$")
#table = olib.Table(X, Xerr, Y, Yerr, title=r"Sandbox plot ", xlabel=r"x ", ylabel=r"y ")
table.printTable()
table.saveAsPDF("Sandbox/test.tex")