import olib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams["text.usetex"] = True


plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif"})
execute = "a"
# 238.a


def a():
    print("Executing a")
    data = pd.read_csv("238_a_Messungen.txt", sep="\t", decimal=",")
    data.columns = [
        r't', 'I_1',	r'U_1',
        r'P_W1',	r'I_2',	r'U_2',
        r'P_W2',	r'P_1',	r'P_2',
        r'f', r'unknown']
    data = data.drop("t", axis=1)
    data = data.drop("f", axis=1)
    data = data.drop("unknown", axis=1)
    data = data.round(2)

    dU = np.zeros_like(data["U_1"])+0.1
    dI = np.zeros_like(data["I_1"])+0.01
    dU_R = np.zeros_like(data["U_2"])+0.1
    dP_1 = np.zeros_like(data["P_1"])+0.01

    data["dU"] = dU
    data["dI"] = dU
    data["dP"] = dP_1

    data2 = pd.DataFrame()

    R = data["U_2"]/data["I_1"]
    dR = ((dU/data["I_1"])**2+(data["U_2"]/data["I_1"]**2*dI)**2)**(1/2)
    data2["R"] = np.round(R, 2)
    data2["dR"] = np.round(dR, 2)

    cos_phi = data["U_2"]/data["U_1"]
    dcos_phi = ((dU_R/data["U_1"])**2 +
                (data["U_2"]/data["U_1"]**2*dU)**2)**(1/2)
    data2["cos_phi"] = np.round(cos_phi, 4)
    data2["d_cos_phi"] = np.round(dcos_phi, 4)

    P_s = data["U_1"]*data["I_1"]
    dP_s = ((data["I_1"]*dU)**2+(data["U_1"]*dI)**2)**(1/2)
    data2["P_s"] = np.round(P_s, 2)
    data2["dP_s"] = np.round(dP_s, 2)

    P_s_cos_phi = data["U_2"]*data["I_1"]
    dP_s_cos_phi = ((data["I_1"]*dU_R)**2+(data["U_2"]*dI)**2)**(1/2)
    data2["P_s_cos_phi"] = np.round(P_s_cos_phi, 2)
    data2["dP_s_cos_phi"] = np.round(dP_s_cos_phi, 2)

    #olib.CSV_to_PDF(data, "238_a_Werte_1.pdf")
    #olib.CSV_to_PDF(data2, "238_b_Werte_2.pdf")

    U_eff = 47
    f = 50
    C = 80e-6
    R_test = 1/(f*np.pi*2*C)
    P_w_max = 1/2*U_eff**2/R_test
    print("R", R_test)
    fig, ax = plt.subplots()
    X, Y = np.array([R, R, R]), np.array([data["P_1"], P_s, P_s_cos_phi])
    Xerr, Yerr = np.array([dR, dR, dR]), np.array(
        [dP_1, dP_s, dP_s_cos_phi])
    ax = olib.setSpace(ax, X, Y, xlabel="$R$[$\Omega$]", ylabel="$P$[W]",
                       title="Abb.1: Leistungen bei einer RC-Schaltung")
    ax, model_P_w = olib.plotData(ax, X[0], Xerr[0], Y[0], Yerr[0],
                                  label=r"$P_w$", color="r", polyfit=0, yscaleing=1, fmt=".", errorbar=False)
    ax, model_P_s = olib.plotData(ax, X[1], Xerr[1], Y[1], Yerr[1],
                                  label=r"$P_s$", color="g", polyfit=0, fmt=".", errorbar=False)
    ax, model_P_s_cos_phi = olib.plotData(ax, X[2], Xerr[2], Y[2], Yerr[2],
                                          label=r"$P_s\cos{\phi}$", color="b", polyfit=0, fmt=".", errorbar=False)
    ax.scatter(R_test, P_w_max, label="$P_{W, max}$")
    plt.legend()
    #plt.show()
    print(P_w_max)
    plt.savefig("238_b_Abb_1", dpi=500)


def c():
    print("Executing c")
    data = pd.read_csv("238_c_Messungen.txt", sep="\t", decimal=",")
    data.columns = [
        r't', r'I_1',	r'U_1',
        r'P_W1',	r'I_2',	r'U_2',
        r'P_W2',	r'P_1',	r'P_2',
        r'f', r'unknown']
    data = data.drop("t", axis=1)
    data = data.drop("f", axis=1)
    data = data.drop("unknown", axis=1)
    data = data.round(2)

    dU = np.zeros_like(data["U_1"])+0.1
    dI = np.zeros_like(data["I_1"])+0.01
    dU_R = np.zeros_like(data["U_2"])+0.1
    dP_1 = np.zeros_like(data["P_1"])+0.01

    data["dU"] = dU
    data["dI"] = dU
    data["dP"] = dP_1
    olib.CSV_to_PDF(data, "238_c_Werte_3.pdf")
    data2 = pd.DataFrame()
    dP_2 = np.zeros_like(data["P_2"])+0.01

    P_S_1 = data["U_1"]*data["I_1"]
    dP_S_1 = ((dU*data["I_1"])**2+(data["U_1"]*dI)**2)**(1/2)
    data2["P_S_1"] = np.round(P_S_1, 2)
    data2["dP_S_1"] = np.round(dP_S_1, 2)

    P_S_2 = data["U_2"]*data["I_2"]
    dP_S_2 = ((dU*data["I_2"])**2+(data["U_2"]*dI)**2)**(1/2)
    data2["P_S_2"] = np.round(P_S_2, 2)
    data2["dP_S_2"] = np.round(dP_S_2, 2)

    P_Cu = data["U_1"]*data["I_1"]+data["U_2"]*data["I_2"]
    dP_Cu = ((dU*data["I_1"])**2+(data["U_1"]*dI)**2 +
             (dU*data["I_2"])**2+(data["U_2"]*dI)**2)**(1/2)
    data2["P_Cu"] = np.round(P_Cu, 2)
    data2["dP_Cu"] = np.round(dP_Cu, 2)

    P_V = data["P_1"]-data["P_2"]
    dP_V = ((dP_1)**2+dP_2**2)**(1/2)
    data2["P_V"] = np.round(P_V, 2)
    data2["dP_V"] = np.round(dP_V, 2)

    P_Fe = P_V-P_Cu
    dP_Fe = (dP_V**2+dP_Cu**2)**(1/2)
    data2["P_Fe"] = np.round(P_Fe)
    data2["dP_Fe"] = np.round(dP_Fe)

    etha = data["P_2"]/data["P_1"]
    dEtha = ((dP_2/data["P_1"])**2+(data["P_2"]/data["P_1"]**2*dP_1)**2)**(1/2)
    data2["eta"] = np.round(etha, 4)
    data2["dEta"] = np.round(dEtha, 4)
    
    olib.CSV_to_PDF(data2, "238_d_Werte_4.pdf")


    Y, X = np.array([data["P_1"], data["P_2"], P_V, P_Cu, P_Fe, etha*1e2]), np.array(
        [data["I_2"], data["I_2"], data["I_2"], data["I_2"], data["I_2"], data["I_2"]])*1e2
    Yerr, Xerr = np.array([dP_1, dP_2, dP_V, dP_Cu, dP_Fe, dEtha]), np.array(
        [dI, dI, dI, dI, dI, dI])

    Yerr, Xerr = np.zeros_like(Yerr), np.zeros_like(Xerr)

    colors = ["#573DC2", "#C2573D", "#3DC257", "#36C9C9", "#C936C9", "#C9C936"]
    titles = ["$P_{W,1}$", "$P_{W,2}$", "$P_{V}$",
              "$P_{Cu}$", "$P_{Fe}$", r"$\eta$"]
    models = [0, 0, 0, 0, 0, 0]


    fig, ax = plt.subplots()

    ax = olib.setSpace(ax, X[:3], Y[:3], "Abb.2: Wirkleistungen in Abhängigkeit der Stromstärke", xlabel=r"$I_2\cdot 10^{2}$[A]", ylabel="$P$[W]")
    ax, models[0] = olib.plotData(ax, X[0], Xerr[0], Y[0], Yerr[0], polyfit=0, color=colors[0], label=titles[0], errorbar=False)
    ax, models[1] = olib.plotData(ax, X[1], Xerr[1], Y[1], Yerr[1], polyfit=0, color=colors[1], label=titles[1], errorbar=False)
    plt.legend()
    #plt.show()
    plt.savefig("238_d_Abb_2", dpi=500)


    fig, ax = plt.subplots()
    ax = olib.setSpace(ax, X[3:5], Y[3:5], "Abb.3: Verlustleistungen Abhängigkeit der Stromstärke", xlabel=r"$I_2\cdot 10^{2}$[A]", ylabel="$P$[W]")
    ax, models[2] = olib.plotData(ax, X[2], Xerr[2], Y[2], Yerr[2], polyfit=0, color=colors[2], label=titles[2], errorbar=False)
    ax, models[3] = olib.plotData(ax, X[3], Xerr[3], Y[3], Yerr[3], polyfit=0, color=colors[3], label=titles[3], errorbar=False)
    ax, models[4] = olib.plotData(ax, X[4], Xerr[4], Y[4], Yerr[4], polyfit=0, color=colors[4], label=titles[4], errorbar=False)
    plt.legend()
    #plt.show()
    plt.savefig("238_d_Abb_3", dpi=500)

    fig, ax = plt.subplots()
    ax = olib.setSpace(ax, X[5], Y[5], "Abb.4: Wirkungsgrad in Abhängigkeit der Stromstärke", xlabel=r"$I_2\cdot 10^{2}$[A]", ylabel=r"$\eta$[\%]")
    ax, models[5] = olib.plotData(ax, X[5], Xerr[5], Y[5], Yerr[5], polyfit=0, color=colors[5], label=titles[5], errorbar=False)
    plt.legend()
    #plt.show()
    plt.savefig("238_d_Abb_4", dpi=500)





def g():
    print("Executing g")
    data = pd.read_csv("238_c_Messungen.txt", sep="\t", decimal=",")
    data.columns = [
        r't', r'I_1',	r'U_1',
        r'P_W1',	r'I_2',	r'U_2',
        r'P_W2',	r'P_1',	r'P_2',
        r'f', r'unknown']
    dU = np.zeros_like(data["U_1"])+0.1
    dI = np.zeros_like(data["I_1"])+0.01
    dP = np.zeros_like(data["P_2"])+0.01

    X = data["I_2"]*1e2
    Xerr = dI*1e2

    quotion = data["U_2"]/data["U_1"]
    Y = quotion*1e2
    dQuotion = ((dU/data["U_1"])**2+(data["U_2"]/data["U_1"]**2*dU))
    Yerr = dQuotion*1e2

    sigma = 0.0225
    dsigma = 0.0018
    omega_L = 613
    domega_L = 79
    R_v = 0.6
    R = data["U_2"]/data["I_2"]
    dR = ((dU/data["I_2"])**2+(data["U_2"]/data["I_2"]**2*dI))

    a = 1-(sigma/2)
    da = 1/2*dsigma
    y = a*R/(R+2*R_v)*(1+(sigma*omega_L/(R+2*R_v))**2)**(-1/2)

    data = pd.DataFrame({"I_2": np.round(data["I_2"], 2), "dI_2": dI, "U_2/U_1": np.round(quotion, 4), "d(U_2/U_1)": np.round(dQuotion, 4), "U_2/U_1 (berechnet)": y})
    olib.CSV_to_PDF(data, "238_g_Werte_5.pdf")

    fig, ax = plt.subplots()
    ax = olib.setSpace(ax, [X, data["I_2"]*1e2], [Y, y*1e2], "Abb.5: Spannungsübertragung",
                       xlabel=r"$I_2\cdot 10^{2}$[A]", ylabel="Verhältniss$\cdot 10^2$")

    ax, models = olib.plotData(
        ax, X, Xerr, Y, Yerr, polyfit=0, errorbar=False, label=r"$\frac{U_2}{U_1}$")
    ax = olib.plotLine(ax, data["I_2"]*1e2, y*1e2, label=r"Berechnet: $\frac{U_2}{U_1}$")

    plt.legend()
    #plt.show()
    plt.savefig("238_g_Abb_5", dpi=500)


def e():
    print("Executing e")
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

    omega_L = data["U_1"][0]/data["I_1"][0]
    domega_L = ((dU[0]/data["I_1"][0])**2+(data["U_1"]
                [0]/data["I_1"][0]**2*dI[0])**2)**(1/2)
    print(f"e: omega_L: {omega_L}±{domega_L}")


def f():
    print("Executing f")
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

    omega_L = data["U_1"][0]/data["I_1"][0]
    domega_L = ((dU[0]/data["I_1"][0])**2+(data["U_1"]
                [0]/data["I_1"][0]**2*dI[0])**2)**(1/2)

    sigma1 = 1-(np.array(data["I_2"])[-1]/np.array(data["I_1"])[-1])**2
    dsigma1 = 2*(np.array(data["I_2"])[-1]/np.array(data["I_1"])[-1])*((1/np.array(data["I_1"])[-1]*dI[0])**2+(np.array(data["I_2"])[-1]/np.array(data["I_1"])[-1]**2*dI[0])**2)**(1/2)
    #dsigma1 = ((2*(1-np.array(data["I_2"])[-1]/np.array(data["I_1"])
    #           [-1])/np.array(data["I_1"])[-1]*dI[0])**2+(2*(1-np.array(data["I_2"])[-1]/np.array(data["I_1"])
                                                          #[-1])*np.array(data["I_2"])[-1]/np.array(data["I_1"])[-1]**2*dI[0])**2)**(1/2)

    sigma2 = 1-(data["U_2"][0]/data["U_1"][0])**2
    dsigma2 = 2*(np.array(data["U_2"])[-1]/np.array(data["U_1"])[-1])*((1/np.array(data["U_1"])[-1]*dU[0])**2+(np.array(data["U_2"])[-1]/np.array(data["U_1"])[-1]**2*dU[0])**2)**(1/2)
    #dsigma2 = ((2*(1-np.array(data["U_2"])[0]/np.array(data["U_1"])
    #            [0])/np.array(data["U_1"])[0]*dI[0])**2+(2*(1-np.array(data["U_2"])[0]/np.array(data["U_1"])
    #                                                    [0])*np.array(data["U_2"])[0]/np.array(data["U_1"])[0]**2*dI[0])**2)**(1/2)

    sigma3 = (np.array(data["U_1"])[-1]/np.array(data["I_1"])[-1]) / (np.array(data["U_1"])[0]/np.array(data["I_1"])[0])
    dsigma3= ((sigma3/np.array(data["U_1"])[-1]*dU[0])**2+(sigma3/np.array(data["I_1"])[-1]*dI[0])**2+(
        sigma3/np.array(data["U_1"])[0]*dU[0])**2+(sigma3/np.array(data["I_1"])[0]*dI[0])**2)**(1/2)

    sigma4= data["U_1"][0]/np.array(data["I_2"])[-1]/omega_L
    dsigma4= ((sigma4/data["U_1"][0]*dU[0])**2+(sigma4/np.array(data["I_2"])[-1]*dI[0])**2+(sigma4/omega_L*domega_L)**2)**(1/2)

    av_sigma = np.mean([sigma1, sigma2, sigma3, sigma4])
    d_av_sigma = ((dsigma1/4)**2+(dsigma2/4)**2+(dsigma3/4)**2+(dsigma4/4)**2)**(1/2)

    print(f"f: s1: {sigma1}±{dsigma1}, s2: {sigma2}±{dsigma2}, s3: {sigma3}±{dsigma3}, s4, {sigma4}±{dsigma4}, av_sigma {av_sigma}±{d_av_sigma}")


if "a" in execute:
    a()
if "c" in execute:
    c()
if "e" in execute:
    e()
if "f" in execute:
    f()
if "g" in execute:
    g()
