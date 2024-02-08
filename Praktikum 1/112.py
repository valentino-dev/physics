import numpy as np
import straight_fit as sf
import math



dT = 0.1
dL = 0.01*10**-3
Temperaturen0 = np.array([np.ndarray(14)+1, np.ndarray(14)+1, np.ndarray(14)+1])

Temperaturen = np.array([range(21, 81, 3), range(21, 81, 3), range(21, 81, 3)])
Längen = np.array([np.array([0.01, 0.02, 0.06, 0.10, 0.13, 0.16, 0.19, 0.23, 0.26, 0.29, 0.33, 0.36, 0.40, 0.43, 0.48, 0.50, 0.52, 0.56, 0.59, 0.62])-0.01, np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])+0.01, [0.00, 0.02, 0.05, 0.07, 0.10, 0.12, 0.14, 0.16, 0.19, 0.21, 0.24, 0.26, 0.29, 0.31, 0.35, 0.36, 0.38, 0.42, 0.44, 0.47]])*10**-3+0.494
Längen0 = 49.4*10**-2

x = Temperaturen-21
y = Längen/Längen0-1
dx = np.zeros(np.shape(x))+1
dy = np.zeros(np.shape(y))
dy = dL*((1/Längen0)**2+(Längen/Längen0**2)**2)**(1/2)

print(np.shape(x), np.shape(dx), np.shape(y), np.shape(dy))
print(x, "+-", dx)
print(np.round(y*10**5, 0), "+-", np.round(dy*10**5, 1))



sf.Plot([np.transpose([x, dx, y, dy] , (1, 0, 2))])
