import numpy as np
import matplotlib.pyplot as plt

# Dieses Fitprogamm wurde von Angelo Valentino Brade im 2. Semester für das p1 geschrieben.


class Fit:

    def __init__(self, x_val=[0], x_err=[0], y_val=[0], y_err=[0], data=[0]):
        # Bei dem Initiieren werden alle Daten für den Fit übergeben. Dabei können die Daten einzelnd in die entsprechenden Arrays übergeben werden, 
        # oder sofern die Syntax bekannt ist in ein mehrdimensionales Array übergeben werden.
        data = np.array(data)
        if np.array_equal(data, [0]):
            self.x_val = np.array(x_val)
            self.x_err = np.array(x_err)
            self.y_val = np.array(y_val)
            self.y_err = np.array(y_err)
        else:
            self.x_val = data[0]
            self.x_err = data[1]
            self.y_val = data[2]
            self.y_err = data[3]
            
        
    def weighted_average(self, z, err):
        return (z/err**2).sum()/(1/err**2).sum()

    def GetFit(self):
        # Die Fitgerade zu den Daten werden berechnet und relevante Ergebnisse übersichtlich ausgegeben.
        y_wa = self.weighted_average(self.y_val, self.y_err)
        xy_wa = self.weighted_average(self.x_val*self.y_val, self.y_err)
        x_wa = self.weighted_average(self.x_val, self.y_err)
        x_sq_wa = self.weighted_average(self.x_val**2, self.y_err)
        
        sigma_wa = self.y_err.size/(1/self.y_err**2).sum()
        V_m = sigma_wa/(self.x_val.size*(x_sq_wa-x_wa**2))
        V_n = sigma_wa/(self.x_val.size*(x_sq_wa-x_wa**2))*x_sq_wa
        m = (xy_wa-x_wa*y_wa)/(x_sq_wa-x_wa**2)
        n = (x_sq_wa*y_wa-x_wa*xy_wa)/(x_sq_wa-x_wa**2)
        
        print("\nFit: ", self, "\n#####################\nm: ", round(m, 7),"+-", round(V_m**(1/2), 12),"; \nn:", round(n, 7), "+-", round(V_n**(1/2), 12), "\n###\nx_wa:", round(x_wa, 12), "\ny_wa:", round(y_wa, 12), "\nxy_wa:", round(xy_wa, 12), "\nx_sq_wa:", round(x_sq_wa, 12), "\nsigma y:", round(sigma_wa, 12), "\nV_m:", round(V_m, 12), "Sigma m: ", round(V_m**(1/2), 12), "V_n", round(V_n, 12), "Sigma n: ", round(V_n**(1/2), 12))
        
        self.m = m
        self.n = n
        self.V_m = V_m
        self.V_n = V_n
        self.x_wa = x_wa
        self.y_wa = y_wa
        self.xy_wa = xy_wa
        self.x_sq_wa = x_sq_wa
        
        return [m, n, V_m**(1/2), V_n**(1/2), x_wa, y_wa, xy_wa, x_sq_wa]   
    
    def predict(self, x):
        return x * self.m + self.n
    
    def GetData(self):
        return [self.x_val, self.x_err, self.y_val, self.y_err]
    
    def PlotFit(self, fit_data, ax=plt, fmt_color=(255, 0, 0), show_error=True, show=True, connect=False, fit=True, ax_label=["x", "y"]):
        # Die Daten werden auf einen gemeinsamen Subplot oder auf eine eigenen geplottet und optional der Fit berechnet. Dafür wird in den Parameter "ax" die Axe des 
        # Subplots übergeben, auf die geplottet werden soll. Wird dafür keine Axe übergeben, so wird von einem plot auf enien einzelnen Subplot ausgegangen.
        # Es kann bei deim Plotten angegeben werden, ob der Plot  gezeigt werden soll, die Daten verbunden werden sollen, gefittet werden soll und welche Bezeichnung die Axen haben sollen.
        fit = np.array(fit)
        
        ax.set_xlabel(ax_label[0])
        ax.set_ylabel(ax_label[1])
        
        x = self.x_val
        y = np.array(fit_data[0]*x+fit_data[1])
        x_positiv_error = self.x_val
        y_positiv_error = (fit_data[0]+fit_data[2])*x+fit_data[1]+fit_data[3]
        x_negativ_error = self.x_val
        y_negativ_error = (fit_data[0]-fit_data[2])*x+fit_data[1]-fit_data[3]
        
        
        if connect:
            ls="-"
        else:
            ls = ""
            
        
        if fit: 
            ax.plot(x, y, linestyle="-", color=fmt_color)
            pass
            if show_error:
                pass
                ax.plot(x_positiv_error, y_positiv_error, linestyle="--", color=fmt_color)
                ax.plot(x_negativ_error, y_negativ_error, linestyle="--", color=fmt_color)
        
        fmt_color = (fmt_color**1.1)
        for i in range(fmt_color.size):
            if fmt_color[i]>1:
                fmt_color[i] = (fmt_color[i]-((fmt_color[i]*255%255))/255)
        ax.errorbar(self.x_val, self.y_val, self.y_err, self.x_err, color=fmt_color, ls=ls, marker="s")
            
        if show:
            plt.show()
            
def PlotFits(fits, ax=plt, show_errors=True, show=True, connect=False, fit=True, ax_label=["x", "y"]):
    # Sollten mehrere Fits auf einen Subplot geplottet werden, so wird für jede Gerade auf einem Subplot eine Farbe gewählt und auf den Subplot geplottet.
    fits = np.array(fits)
    colors = np.array([(153, 95, 95), (153, 148, 95), (100, 153, 95), (95, 153, 152), (95, 99, 153), (153, 95, 150), (128, 95, 153), (153, 124, 95)]) / 255
    if (colors.size < fits.size):
        print("ERROR: To many plots! (max. 8)")
        return
    for i in range(fits.size):
        fits[i].PlotFit(fits[i].GetFit(), ax=ax, fmt_color=colors[i], show=False, connect=connect, fit=fit, ax_label=ax_label)
    if show:
        plt.show()
    
def FusePlots(plots, show=True, connect=False, fit=True, ax_label=[["x", "y"]]):
    # Sollten mehrere Subplots benötigt werden, so werden diese hier kreiert und zusammengefügt.
    plots = np.array(plots)
    ax_label = np.array(ax_label)
    
    # Da connect ein Boolean oder ein Array aus Boolean sein darf, muss hier festgestellt werden, was es ist.
    if isinstance(connect, bool):
        connect_temp = np.zeros(np.shape(plots)[0])
        if connect:
            connect_temp = connect_temp+1
            
        connect = connect_temp
    if isinstance(fit, bool):
        fit_temp = np.zeros(np.shape(plots)[0])
        if fit:
            fit_temp = fit_temp+1
        fit = fit_temp
        

    fig, axs = plt.subplots(plots.shape[0])
    if plots.shape[0] == 1:
        axs = [axs]
    for i in range(plots.shape[0]):
        PlotFits(fits=plots[i], ax=axs[i], show=False, connect=connect[i]==1, fit=fit[i]==1, ax_label=ax_label[i])
    
    if show:
        plt.show()
    
def Plot(data, show=True, connect=False, fit=True, ax_label=[["x", "y"]]):
    # Üblicherweise wird der Nutzer nur diese Funktion verwenden, da alle Aufgaben von dieser übernommen werden und die Syntax immer die gleiche ist.
    # In diese Funktionen können alle Daten übergeben werden, die automatischa alle Subplots und fits erstellt. Welche fits zusammen auf einen Subplot kommen wird 
    # durch die Shape ausgemacht. Ferner kann folgend die Syntax, in der die Daten übergeben werden MÜSSEN betrachtet werden:
    # data = [plots = [fits = [data = [x, xerr, y, yerr]]]]
    # Bei dem Auswertern vor dem Plotten sind oft die Daten noch im falschen Vormat. Eine gängige Lösung bietet die Funktion np.tanspose. Ferner ist oft der Parameter "[np.transpose([x, dx, y, dy] , (1, 0, 2))]".
    data = np.array(data)
    fits = np.ndarray((np.shape(data)[0], np.shape(data)[1]), dtype="object")
    for i in range(data.shape[0]):
        for k in range(data.shape[1]):
            fits[i][k] = Fit(data=data[i][k])
    
    FusePlots(fits, show=False, connect=connect, fit=fit, ax_label=ax_label)
    
    if show:
        plt.show()
        pass
    else:
        return fits
            

        
        
        
    
    