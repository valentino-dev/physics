import numpy as np
import matplotlib.pyplot as plt
import geradenfit as gf

# Anzahl Messungen (i.e. wie viele Plots erstellt werden müssen / wie viele verschiedene Messungen gemacht worden sind)

class AM:
    anzahl_messungen = 2 #<---

# Konstanten

g = 9.81 #m/s**2

# Messewrte
#   Messwerte: aufgabeMessung = np.array([]) #Einheit
#   Fehler: d_aufgabeMessung = np.zeros(len(Messwerte))+Fehler / np.array([]) #Einheit

aGewicht = np.array([10.0,20.0,30.0,40.0,50.0]) #kg
d_aGewicht = np.zeros(len(aGewicht))+1.5 #kg

bGewicht = np.array([10,20,30,40,50,60,70,80,90,100]) #kg
bGeschw = np.array([12,14,15,18,29,33,34,35,36,37]) #m/s
d_bGeschw = np.zeros(len(bGeschw))+4 #m/s

# Rechnung
#   Messwerte: aufgabeGröße = Rechnung #Einheit
#   Fehler: d_aufgabeGröße = Rechnung #Einheit

aKraft = aGewicht*g
d_aKraft = d_aGewicht*g

# Achsen
#   x =
#   y =
#   yerr =

class Achsen:
    x = [aGewicht,bGewicht] #<---
    y = [aKraft,bGeschw] #<---
    yerr = [d_aKraft,d_bGeschw] #<---
    color = 'blue'
