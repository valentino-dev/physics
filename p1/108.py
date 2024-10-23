
dZeit = np.array([a25, a50, a75, a100])
dPerioden_arr = np.repeat([dPerioden], 4, axis=0)

dPerioden0 = np.array([5, 6, 7, 8, 9, 11, 12, 13, 14, 15])
a0 = np.array([6.0, 7.2, 8.5, 9.7, 11.0, 13.4, 14.7, 16.0, 17.2, 18.4])


######--108.a


aAuslenkung_Korregiert = np.ndarray((3, 8))
for i in range(aAuslenkung0.size):
    aAuslenkung_Korregiert[i] = aAuslenkung0[i]-aAuslenkung[i]
   
aKraft = aGewicht*9.81

######--108.b
bAuslenkung_Korregiert = np.ndarray((3, 8))
for i in range(bAuslenkung0.size):
    bAuslenkung_Korregiert[i] = bAuslenkung[i]-bAuslenkung0[i]

bKraft = bGewicht*9.81

######--108.d
dPeriodendauer = dZeit/dPerioden_arr
dDelta_Periodendauer = delta_Zeit/dPerioden_arr
print("T", np.round(dPeriodendauer, 3), "+-", np.round(dDelta_Periodendauer, 3))
dPeriodendauer0 = a0/dPerioden0
dDelta_Periodendauer0 = delta_Zeit/dPerioden0
print("T0", np.round(dPeriodendauer0, 3), "+-", np.round(dDelta_Periodendauer0, 3))
#dDelta_Periodendauer_gesammt = np.apend(dDelta_Periodendauer, dDelta_Periodendauer0, axis=0)
dPeriodendauer_gemittelt = np.zeros(5)
dDelta_Periodendauer_gemittelt = np.zeros(5)
for i in range(4):
    dPeriodendauer_gemittelt[i+1] = dPeriodendauer[i].sum()/dPeriodendauer[i].size
    dDelta_Periodendauer_gemittelt[i+1] = (dDelta_Periodendauer[i]**2).sum()/dDelta_Periodendauer[i].size**2

dPeriodendauer_gemittelt[0] = dPeriodendauer0.sum()/dPeriodendauer0.size
dDelta_Periodendauer_gemittelt[0] = (dDelta_Periodendauer0**2).sum()/dDelta_Periodendauer0.size**2
dDelta_Periodendauer_gemittelt = dDelta_Periodendauer_gemittelt**(1/2)
print("aP", np.round(dPeriodendauer_gemittelt, 3), "+-", np.round(dDelta_Periodendauer_gemittelt, 3))


dPeriodendauer_sq = dPeriodendauer_gemittelt**2
dDelta_Periodendauer_sq = dDelta_Periodendauer_gemittelt*dPeriodendauer_gemittelt*2


dAbstände_sq = dAbstände**2
print("P_sq", np.round(dPeriodendauer_sq, 3), "+-", np.round(dDelta_Periodendauer_sq, 3), "; a_sq", dAbstände_sq)

dDelta_Abstände_sq = np.zeros(5)

#print(dPeriodendauer_gemittelt)
#print(np.shape(dPeriodendauer_sq), np.shape(dAbstände_sq), np.shape(dDelta_Abstände_sq), np.shape(dDelta_Periodendauer_sq))

#a und b
#sf.Fit.Plot([np.transpose([ aKraft, aDelta_Gewicht, aAuslenkung_Korregiert, aDelta_Auslenkung_Array], (1, 0, 2)), np.transpose([bKraft, bDelta_Gewicht, bAuslenkung_Korregiert, bDelta_Auslenkung_Arr], (1, 0, 2))], connect=[0, 1, 0], fit=[1, 0, 1])

#d
#print("Abstände_sq", 10/0.01*dAbstände_sq, "+-", dDelta_Abstände_sq, "Tsq", 15/9*dPeriodendauer_sq, "+-", dDelta_Periodendauer_sq)
sf.Fit.Plot([[[dAbstände_sq, dDelta_Abstände_sq, dPeriodendauer_sq, dDelta_Periodendauer_sq]]])
aSkale = 0
bSkale = 5/15
bSkalex = 16/80


print("aKraft: ", str(np.round(aKraft, 3))+"+-"+str(np.round(aDelta_Gewicht, 3)))
print("aAuslenkung_Korregiert: ", str(np.round(aAuslenkung_Korregiert, 6))+"+-"+str(np.round(aDelta_Auslenkung_Array, 5)))
print("bKraft: ", str(np.round(bKraft*bSkalex, 2))+"+-"+str(np.round(bDelta_Gewicht, 2)))
print("bAuslenkung_Korregiert: ", str(np.round(bAuslenkung_Korregiert*bSkale, 6))+"+-"+str(np.round(aDelta_Auslenkung_Array, 5)))
print("dAbstände_sq: ", str(np.round(dAbstände_sq[:], 7))+"+-"+str(np.round(dDelta_Abstände_sq[:], 5)))
print("Periodendauer: ", str(np.round(dPeriodendauer_sq[:], 2))+"+-"+str(np.round(dDelta_Periodendauer_sq[:], 2)))

m = np.array([2.718, 1.578, 1.891])*10**-3
delta_m = np.array([0.031, 0.031, 0.031])*10**-3
l = np.array([395, 395, 395])*10**-3
b = np.array([10.54, 10.23, 10.12])*10**-3
delta_l = 0.5*10**-3
delta_b = 0.05*10**-3
delta_h = 0.002*10**-3
h = np.array([2.043, 2.011, 1.525])*10**-3
dI = 1/12*((delta_b*h**3)**2+(3*b*h**2*delta_h)**2)**(1/2)
I = 1/12*b*h**3
dE = l**3/48/I/m*((3*delta_l/l)**2+(delta_h/h)**2+(delta_m/m)**2)**(1/2)
E = l**3/48/I/m
print(I, "+-", dI,"; ", E, "+-", dE)


F0 = np.array([43, 25, 76])
delta_F0 = np.array([2, 2, 2])
l = np.array([400, 400, 400])*10**-3
delta_l = np.array([0.1, 0.1, 0.1])*10**-3
b = np.array([10.12, 20.00, 20.42])*10**-3
delta_b = np.array([0.005, 0.005, 0.005])*10**-3
h = np.array([1.525, 4.16, 3.124])*10**-3
delta_h = np.array([0.002, 0.002, 0.002])*10**-3


dI = 1/12*((delta_b*h**3)**2+(3*b*h**2*delta_h)**2)**(1/2)
I = 1/12*b*h**3
E=l**2*F0/I/np.pi**2
dE = E*((2*delta_l/l)**2+(delta_F0/F0)**2+(dI/I)**2)**(1/2)


print(I, "+-", dI,"; ", E, "+-", dE)