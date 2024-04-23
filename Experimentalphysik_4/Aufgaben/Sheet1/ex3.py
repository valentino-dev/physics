import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append("/home/angelo/Git/olib")
import olib


def a():
    k = 1.38e-23
    m = 1.66e-27
    h = 6.62e-34
    X = np.linspace(50e-9, 1e-3, 100)
    Y = (h / (2 * m * k * X) ** (1 / 2)) ** -3
    fig, ax = plt.subplots()
    table = olib.Table(
        X,
        Y,
        title=r"Fig. 1: $d=\lambda_{dB}$ at $n=1.25\cdot 10^{19}$[$\frac{1}{\text{m}^3}$]",
        xlabel=r"$T$ [K]",
        ylabel=r"$n$[$\frac{1}{\text{m}^3}$]",
    )
    ax, _ = olib.plotData(ax, table, polyfit=0)
    ax = olib.setSpace(ax, table)
    plt.savefig("Fig1.png", dpi=500)


a()
