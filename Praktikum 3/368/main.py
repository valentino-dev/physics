import olib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
exec = "f"

def b():
    x_m = np.array([2680, 2490, 2339, 2158, 1984, 1807, 1652, 1473, 1306])*1e-5
    x_m_0 = 2886e-5
    X = x_m_0-x_m
    dx_m = 10*1e-5
    m = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    f = 29.2*1e-2
    df = 0.5*1e-2
    D = 0.1e-3
    dD = 0.001e-3
    
    table = olib.Table(m, np.zeros_like(m), X, np.zeros_like(x_m)+dx_m, r"368.b: Minima am Einfachspalt", r"$m$ [Ordung]", r"$x_m$[m]")
    
    fig, ax = plt.subplots()
    
    ax, model = olib.plotData(ax, table, fmt="v")
    ax = olib.setSpace(ax, table, path="")
    model.printParameter()
    Lambda = model.m/f*D
    dLambda = ((model.V_m**(1/2)/f*D)**2+(model.m/f**2*df*D)**2+(model.m/f*dD)**2)**(1/2)
    print(Lambda, dLambda)
    table.saveAsPDF()
    
def c():
    g=10.4e-2
    dg=0.2e-2
    b=61e-2
    db = 1e-2
    B=1e-2
    dB=0.2e-2
    print(olib.roundToError(np.array([g/b*B], np.array([((dg/b*B)**2+(g/b**2*db*B)**2+(g/b*dB)**2)**(1/2)]))))
    
def d():
    alpha = (np.array([5555, 6115, 6670, 7235, 7800, 8360, 8930, 9500, 4440, 3885, 3330, 2770, 2215, 1655, 1090, 530])*1e-4)
    phi_m = (alpha[0:8]-alpha[8:])/2
    dalpha = (np.zeros_like(alpha)+0.1e-2)
    dphi_m = ((dalpha[0:8]/2)**2+(dalpha[8:]/2)**2)**(1/2)
    print(phi_m)
    
    m = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    Lambda = 546.07e-9
    g = m*Lambda/2/np.sin(phi_m/2)
    dg = m*Lambda/2/np.sin(phi_m/2)**2*np.cos(phi_m/2)/2*dphi_m
    print(g.mean(), np.linalg.norm(dg)/dg.shape[0])
    table = olib.Table(phi_m, dphi_m, g, dg, "Gitterkonstante", r"$phi_m$[Bogenmass]", r"$g$[m]")
    table.saveAsPDF(height=-1)
    
def e():
    alpha = np.array([4555, 4110, 3665, 3220, 2780, 2330, 1880, 1440, 5440, 5890, 6330, 6780, 7230, 7680, 8130, 8585])*1e-4
    phi_m = (alpha[8:]-alpha[:8])/2
    dalpha = (np.zeros_like(alpha)+0.1e-2)
    dphi_m = ((dalpha[0:8]/2)**2+(dalpha[8:]/2)**2)**(1/2)
    g = 9.811e-6
    dg = 1.9e-8
    m = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    Lambda = np.sin(phi_m/2)*2*g/m
    dLambda = ((np.cos(phi_m/2)*dphi_m*g/m)**2+(Lambda/g*dg)**2)**(1/2)
    print(Lambda.mean(), np.linalg.norm(dLambda)/dLambda.shape[0])
    table = olib.Table(phi_m, dphi_m, Lambda, dLambda, "Wellenlänge", r"$phi_m$[Bogenmass]", r"$\lambda$[m]")
    table.saveAsPDF(height=-1)
    
def f():
    
    g_G = 9.811e-6
    dg_G = 1.9e-8
    B = np.zeros(3)+0.2e-2
    dB = np.array([0.05, 0.05, 0.05])*1e-2
    b = (np.array([114, 111, 108])+163.5)*1e-2
    db = np.array([0.3, 0.3, 0.3])*1e-2
    g = np.array([6, 6, 6])*1e-2
    dg = np.array([0.1, 0.1, 0.1])*1e-2
    m = np.array([2, 4, 7])
    
    
    G = B/b*g
    dG = ((dB/b*g)**2+(B/b**2*db*g)**2+(B/b*dg)**2)**(1/2)
    N=G/g_G
    dN = ((dG/g_G)**2+(G/g_G**2*dg_G)**2)**(1/2)
    
    A = m*N
    dA = m*dN
    print(G, dG)
    print(A, dA)
    print(N, dN)
    print(A.mean(), np.linalg.norm(dA)/dA.shape[0])
    


for char in exec:
    locals()[char]()