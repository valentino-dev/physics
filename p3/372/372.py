import numpy as np


def a():

    t3 = np.arange(10,190,10)

    U3 = np.array([.1,.2,.5,.6,.6,.8,.8,.7,.7,.8,.8,.7,.7,.8,.9,.8,.7,.8])+2

    DU = 0.1

    for i in range(len(t3)):
        print(t3[i],U3[i],DU)

def a2():

    t5 = np.arange(10,310,10)

    U5 = np.array([90.0,129.0,140.0,146.0,149.0,151.0,154.0,155.9,157.4,159.0,160.2,161.3,162.3,163.1,163.8,164.6,165.2,165.7,166.2,166.8,167.3,167.7,168.1,168.5,168.9,169.4,169.6,170.0,170.3,170.5])

    DU = 0.1

    for i in range(len(t5)):
        print(t5[i],U5[i],DU)

def b():

    T = np.array([29,33,38,44,48,53,59,64,69,75])+273.15
    T0 = 25+273.15
    T4 = T**4-T0**4

    U_schwarz = np.array([15.0,19.4,25.9,31.1,38.0,44.4,51.6,58.6,66.3,73.4])
    DU_schwarz = 0.1
    U_matt = np.array([4.7,7.0,7.0,8.0,9.6,10.7,12.9,13.8,15.9,16.9])
    DU_matt = np.array([1,1,1,.1,.1,.1,.1,.1,.1,.1])
    U_weiß = np.array([17.5,21.0,26.0,31.9,38.5,44.9,51.9,59.4,66.3,74.5])
    DU_weiß = np.array([1,1,1,.1,.1,.1,.1,.1,.1,.1])
    U_glänzend = np.array([4.5,6.0,4.7,5.5,6.5,6.6,8.2,8.7,10.5,11.1])
    DU_glänzend = np.array([2,2,2,.1,.1,.1,.1,.1,.1,.1])

    V = 100
    S = 29 # µV m² 1/W
    A = 10.4*9.9
    DA = ((0.1*9.9)**2+(0.1*10.4)+*2)**0.5

    PHI/F = 


def c():

    r = np.arange(710,210,-50)-880
    Dr = 1

    U = np.array([840.0,495.0,319.7,215.6,152.7,111.9,85.3,66.4,53.0])
    DU = np.array([1,1,.1,.1,.1,.1,.1,.1,.1])

def c2():

    UH = np.array([12,10.6,8.8,7.1,5.7,4.4,3.2,2.2,1.2])
    I = np.array([4.34,4.02,3.63,3.20,2.81,2.41,2.00,1.61,1.21,.8])

    U = np.array([1080.0,1517.0,1160.0,833.0,585.0,401.0,240.0,130.0,60.0,29.0])
    DU = np.array([10,10,1,1,1,1,1,1,1,1])

a2()