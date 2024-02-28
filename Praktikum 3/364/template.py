import olib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Latex font
plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif"})

# Example function for calculating resulsts and errorfrom observibles
def calc(x, xerr):
    y = x*np.cos(x)+x
    yerr = ((xerr*np.cos(x))**2+(x+np.cos(x)*xerr)**2+(xerr)**2)**(1/2)
    return y, yerr

# Path to Data and Error
PATH_DATA = ""
PATH_ERR = ""
#data = pd.read_csv(PATH_DATA, header=None, sep=";").to_numpy()
#data_err = pd.read_csv(PATH_ERR, header=None, sep=";").to_numpy()

# Example Data
data = np.linspace(0, 100, 10)+1
data_err = np.linspace(0, 100, 10)*1e-2

# Defining X and Y
X, Xerr = data, data_err
Y, Yerr = calc(data, data_err)

# Defining Subplots for olib
fig, ax = plt.subplots()

# Printing (with fmt=","), Fitting (with polyfit=1; no fit with polyfit=0 and fit on n polynomial with polyfit=n) and Predicting (Automaticly with fitting) Data
ax, model = olib.plotData(ax, X, Xerr, Y, Yerr, label="Template Data", polyfit=1)

# Defining print Space with grid lines, labels and etc.
ax = olib.setSpace(ax, X, Y, "Template", xlabel="x Label", ylabel="y Label")

# Print the parameters of the fittet model
model.printParameter()

# Show plot
plt.show()

# Print Plot
# plt.save("Example Path/fig.pdf", dpi=500)
