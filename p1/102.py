import numpy as np
import math
import matplotlib.pyplot as plt
from straight_fit import fit






dphi=0.5
phi1=np.array([13, 10.8, 7.8])
phi2=np.array([4.8, 1.4, 1.2])
n1=np.array([1, 1, 1])
n2=np.array([15, 8, 4])

print(np.exp(-(np.log(phi1)-np.log(phi2))/(n1-n2))/(n2-n1)*((dphi/phi1)**2+(dphi/phi2)**2)**(1/2))

print(np.exp(-(np.log(phi1)-np.log(phi2))/(n2-n1)))
print(np.exp(-(np.log(phi1)-np.log(phi2))/(n2-n1))*((dphi/phi1*(n1-n2))**2+(dphi/phi2*(n1-n2))**2)**(1/2))
print("dQ", (((n1-n2)*math.pi*dphi/(phi1*(np.log(phi1)-np.log(phi2))**2))**2+((n1-n2)*math.pi*dphi/(phi2*(np.log(phi1)-np.log(phi2))**2))**2)**(1/2))


phi11=np.array([13, 12.6, 12, 11.2, 10.7, 9.8, 9.2, 8.7, 8.4, 7.7, 7.1, 6.5, 5.8, 5.2, 4.8])
phi12=np.array([10.8, 8.8, 7.2, 5.4, 4, 3, 2.1, 1.4, 0.8, 0.2])
phi13=np.array([7.8, 4.4, 2.4, 1.2, 0.4, 0.1])


phi21=np.array([0.9, 1, 1.1, 1.2, 1.7, 3, 4, 2.2, 1.1, 0.9, 0.8, 2.5, 2.8, 4.2, 4.1])
phi22=np.array([0.9, 0.9, 1, 1.2, 1.5, 1.8, 1.8, 1.3, 1, 0.8, 0.7, 1.6, 1.7, 1.8, 1.8])



print("1: ", dphi/phi11)
print("2: ", dphi/phi12)
print("3: ", dphi/phi13)


do=0
if(do==0):
    path="C:\\Users\\angel\\Meine Ablage\\Studium\\Studium Physik\\Praktikum 1\\102vb.txt"
    f = open(path, "r")
    content = f.read()
    f.close()

    #print(content)
    size = 15
    rows = 3
    arr1 = [""]*size
    arr2 = [""]*size
    arr3 = [""]*size

    k = 0
    h = 0
    lastd = "\n"
    for i in range(len(content)):
        if(content[i]==" "):
            lastd = " "
            k+=1
        elif(content[i]=="\n"):
            lastd = "\n"
            k+=1
        else:
            if(k%rows==0):
                arr1[int(k/rows)]+=content[i]
            elif(k%rows==1):
                arr2[int(k/rows)]+=content[i]
            elif(k%rows==2):
                arr3[int(k/rows)]+=content[i]
                
    
elif(do==1):
    
    path="102vd.txt"
    f = open(path, "r")
    content = f.read()
    f.close()
    #print(content)
    size = 4
    rows = 4
    arr1 = [""]*size
    arr2 = [""]*size
    arr3 = [""]*size
    arr4 = [""]*size
    

    k = 0
    h = 0
    lastd = "\n"
    for i in range(len(content)):
        if(content[i]==" "):
            lastd = " "
            k+=1
        elif(content[i]=="\n"):
            lastd = "\n"
            k+=1
        else:
            if(k%rows==2):
                arr1[int(k/rows)]+=content[i]
            elif(k%rows==1):
                arr2[int(k/rows)]+=content[i]
            elif(k%rows==0):
                arr3[int(k/rows)]+=content[i]
            elif(k%rows==3):
                arr4[int(k/rows)]+=content[i]
    
    arr1 = np.array(arr1, dtype="f")
    arr2 = np.array(arr2, dtype="f")
    arr3 = np.array(arr3, dtype="f")
    arr4 = np.array(arr4, dtype="f")
    print(arr1, arr2, arr3, arr4)
    
            
#arr1 = np.array(arr1, dtype="f")
#arr2 = np.array(arr2, dtype="f")
#arr3 = np.array(arr3, dtype="f")

#arr1 = arr1-1.4
#arr2 = arr2-1.4
#arr3 = arr3-1.4

#x_val_1 = np.log10(arr1)
#x_val_2 = np.log10(arr2)
#x_val_3 = np.log10(arr3)

#x_val_1 = x_val_1[(x_val_1>=0)]
#x_val_2 = x_val_2[(x_val_2>=0)]
#x_val_3 = x_val_3[(x_val_3>=0)]

#arr1 = arr1[:x_val_1.size]
#arr2 = arr2[:x_val_2.size]
#arr3 = arr3[:x_val_3.size]


phy_err=0.5
fitting = fit()
def fit_arr(arr):
    arr = np.array(arr, dtype="f")
    arr = arr - 1.4
    x_val = arr[(arr>=1)]
    x_val = np.log(x_val)
    arr = arr[:x_val.size]
    print("err", ((1/arr)*phy_err))
    return fitting.calc_m_n_Vm_Vn(x_val=np.arange(1, x_val.size+1), x_err=np.zeros(x_val.size), y_val=x_val, y_err=((2/arr)*phy_err)), x_val
    #return fitting.calc_m_n_Vm_Vn(x_val=np.arange(1, x_val.size+1), x_err=np.zeros(x_val.size), y_val=x_val, y_err=((1/arr)*phy_err)), x_val


def do1():
    arr1_fit=fit_arr(arr1)
    arr2_fit=fit_arr(arr2)
    arr3_fit=fit_arr(arr3)

    data1=arr1_fit[0]
    y_v_1=arr1_fit[1]
    x_v_1=np.arange(1, y_v_1.size+1)
    data2=arr2_fit[0]
    y_v_2=arr2_fit[1]
    x_v_2=np.arange(1, y_v_2.size+1)
    data3=arr3_fit[0]
    y_v_3=arr3_fit[1]
    x_v_3=np.arange(1, y_v_3.size+1)
    #print(x_v_2)
    print(data1)
    #print(data2)
    #print(data3)

    #plt.ylim(0, 2)

    plt.plot(x_v_1, y_v_1, "r", x_v_1, data1[1]+data1[0]*x_v_1, "r--", x_v_2, y_v_2, "g",  x_v_2, data2[1]+data2[0]*x_v_2, "g--", x_v_3, y_v_3, "b",  x_v_3, data3[1]+data3[0]*x_v_3, "b--")
    #plt.plot(range(15), arr1, "r", range(15), arr2, "g", range(15), arr3, "b")
    #plt.ylabel('some numbers')
    plt.show()
    
def do2():
    
    fitted=fitting.calc_m_n_Vm_Vn(arr1, np.zeros(arr2.size), arr3, arr4)
    print(fitted)
    x_f=np.arange(np.amax(arr1))
    plt.ylim(0, 1)
    plt.plot(arr1, arr3, "r", x_f, fitted[1]+fitted[0]*x_f, "b--")
    plt.show()

if(do==0):
    do1()
elif(do==1):
    do2()