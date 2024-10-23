from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import olib

X = np.array(range(20))/10
Y = np.sin(X) * np.random.rand(20) *2 + X
X_err = 1/(X+1)*0.2
Y_err = 1/(Y+1)*0.2

mode = 1
if mode == 0:
    
    model1 = LinearRegression().fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    model2 = LinearRegression().fit(X.reshape(-1, 1), Y.reshape(-1, 1), 1/X_err)
    plt.plot(X, np.squeeze(model1.predict(X.reshape(-1, 1))), label="unweighted")
    plt.plot(X, np.squeeze(model2.predict(X.reshape(-1, 1))), label="weighted")
    plt.errorbar(X, Y, Y_err, xerr=X_err, fmt="x")
    plt.legend()
    plt.show()
    
elif mode == 1:
    fig, ax = plt.subplots()
    ax = olib.setSpace(ax, X, Y)
    ax = olib.plotData(ax, X, X_err, Y, Y_err)
    plt.show()
