import olib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams["text.usetex"] = True


plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif"})
execute = ["c"]
# 238.a


def a():
    print("Executing a")
    data = pd.read_csv("238_a_Messungen.txt", sep="\t", decimal=",")
    data.columns = [
        r't', r'I_1',	r'U_1',
        r'P_W1',	r'I_2',	r'U_2',
        r'P_W2',	r'P_1',	r'P_2',
        r'f', r'unknown']
    dU = np.zeros_like(data["U_1"])+0.1
    dI = np.zeros_like(data["I_1"])+0.01
    dU_R = np.zeros_like(data["U_2"])+0.1
    dP_1 = np.zeros_like(data["P_1"])+0.01

    R = data["U_2"]/data["I_1"]
    dR = ((dU/data["I_1"])**2+(data["U_2"]/data["I_1"]**2*dI)**2)**(1/2)

    cos_phi = data["U_2"]/data["U_1"]
    dcos_phi = ((dU_R/data["U_1"])**2 +
                (data["U_2"]/data["U_1"]**2*dU)**2)**(1/2)

    P_s = data["U_1"]*data["I_1"]
    dP_s = ((data["I_1"]*dU)**2+(data["U_1"]*dI)**2)**(1/2)

    P_s_cos_phi = data["U_2"]*data["I_1"]
    dP_s_cos_phi = ((data["I_1"]*dU_R)**2+(data["U_2"]*dI)**2)**(1/2)

    fig, ax = plt.subplots()
    X, Y = np.array([R, R, R]), np.array([data["P_1"], P_s, P_s_cos_phi])
    Xerr, Yerr = np.array([dR, dR, dR]), np.array(
        [dP_1, dP_s, dP_s_cos_phi])
    ax = olib.setSpace(ax, X, Y, xlabel="$R$[$\Omega$]", ylabel="$P$[W]",
                       title="Abb.1: Leistungen bei einer RC-Schaltung")
    ax, model_P_w = olib.plotData(ax, X[0], Xerr[0], Y[0], Yerr[0],
                                  label=r"$P_w$", color="r", polyfit=0, yscaleing=1)
    ax, model_P_s = olib.plotData(ax, X[1], Xerr[1], Y[1], Yerr[1],
                                  label=r"$P_s$", color="g", polyfit=0)
    ax, model_P_s_cos_phi = olib.plotData(ax, X[2], Xerr[2], Y[2], Yerr[2],
                                          label=r"$P_s\cos{\phi}$", color="b", polyfit=0)
    plt.legend()
    plt.show()


def c():
    print("Executing c")
    data = pd.read_csv("238_c_Messungen.txt", sep="\t", decimal=",")
    data.columns = [
        r't', r'I_1',	r'U_1',
        r'P_W1',	r'I_2',	r'U_2',
        r'P_W2',	r'P_1',	r'P_2',
        r'f', r'unknown']
    dU = np.zeros_like(data["U_1"])+0.1
    dI = np.zeros_like(data["I_1"])+0.01
    dU_R = np.zeros_like(data["U_2"])+0.1
    dP_1 = np.zeros_like(data["P_1"])+0.01
    dP_2 = np.zeros_like(data["P_2"])+0.01

    P_S_1 = data["U_1"]*data["I_1"]
    dP_S_1 = ((dU*data["I_1"])**2+(data["U_1"]*dI)**2)**(1/2)

    P_S_2 = data["U_2"]*data["I_2"]
    dP_S_2 = ((dU*data["I_2"])**2+(data["U_2"]*dI)**2)**(1/2)

    P_Cu = data["U_1"]*data["I_1"]+data["U_2"]*data["I_2"]
    dP_Cu = ((dU*data["I_1"])**2+(data["U_1"]*dI)**2 +
             (dU*data["I_2"])**2+(data["U_2"]*dI)**2)**(1/2)

    P_V = data["P_1"]-data["P_2"]
    dP_V = ((dP_1)**2+dP_2**2)**(1/2)

    P_Fe = P_V-P_Cu
    dP_Fe = (dP_V**2+dP_Cu**2)**(1/2)

    etha = data["P_2"]/data["P_1"]
    dEtha = ((dP_2/data["P_1"])**2+(data["P_2"]/data["P_1"]**2*dP_1)**2)**(1/2)

    X, Y = np.array([data["P_1"], data["P_2"], P_V, P_Cu, P_Fe, etha]), np.array([data["I_2"], data["I_2"], data["I_2"], data["I_2"], data["I_2"], data["I_2"]])

    fig, ax = plt.subplots()
    ax = S




if "a" in execute:
    a()
if "c" in execute:
    c()
