import numpy as np
import matplotlib.pyplot as plt
import auswertung as aw

class Gerade:

    def __init__(self):
        pass

    def mnvmvn(self,x,y,yerr):

        print('----------')

        x_bar = ((x/yerr**2)).sum()/(1/yerr**2).sum()
        print('x_bar',x_bar)

        x_bar_sq = x_bar**2
        print('x_bar_sq',x_bar_sq)

        x_sq_bar = ((x**2)/(yerr**2)).sum()/(1/yerr**2).sum()
        print('x_sq_bar',x_sq_bar)

        y_bar = (y/yerr**2).sum()/(1/yerr**2).sum()
        print('y_bar',y_bar)

        xy_bar = ((x*y)/(yerr)**2).sum()/(1/(yerr)**2).sum()
        print('xy_bar',xy_bar)

        sigma_sq = x.size/(1/yerr**2).sum()
        print('sigma_sq',sigma_sq)

        m = (xy_bar-x_bar*y_bar)/(x_sq_bar-x_bar_sq)
        print('m',m)

        n = (x_sq_bar*y_bar-x_bar*xy_bar)/(x_sq_bar-x_bar_sq)
        print('n',n)

        v_m = (sigma_sq/(x.size*(x_sq_bar-x_bar_sq)))**0.5
        print('sigma_m',v_m)

        v_n = ((sigma_sq*x_sq_bar)/(x.size*(x_sq_bar-x_bar_sq)))**0.5
        print('sigma_n',v_n)

        print('----------')

        return m,n,v_m,v_n

class Plot:

    def __init__(self):
        pass

    def ausgleichsgerade(self,steigung,x,b):
        fit = steigung*x+b

        return fit 

    def plot(self,x,y,yerr,fit,color):
        plt.errorbar(x,y,yerr,fmt=color,ls="",marker="d")
        plt.plot(x,fit,color)
        plt.show()

class Auswertung:

    def __init__(self):
        pass

    def auswertung(self):
        i_AM = aw.AM()
        i_Achsen = aw.Achsen()
        i_Gerade = Gerade()
        i_Plot = Plot()

        x = i_Achsen.x
        y = i_Achsen.y
        yerr = i_Achsen.yerr
        color = i_Achsen.color

        anzahl_messungen = i_AM.anzahl_messungen
        mnvmvn = np.ndarray(shape=(anzahl_messungen,4),dtype=float)

        for i in range(anzahl_messungen):
            mnvmvn[i] = i_Gerade.mnvmvn(x[i],y[i],yerr[i])
            steigung = mnvmvn[i][0]
            b = mnvmvn[i][1]
            fit = i_Plot.ausgleichsgerade(steigung,x[i],b)
            i_Plot.plot(x[i],y[i],yerr[i],fit,color)

i_Auswertung = Auswertung()
i_Auswertung.auswertung()
