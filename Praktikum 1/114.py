
k=[0, 1, 2, 3, 4, 5, 6, 7]
n=[51, 82, 90, 41, 22, 7, 6, 1]
nk=[0, 0, 0, 0, 0, 0, 0, 0]
dnk=[0, 0, 0, 0, 0, 0, 0, 0]
p=[0, 0, 0, 0, 0, 0, 0, 0]
N = 0

for i in range(len(k)):
    N += n[i]

sum = 0
for i in range(len(k)):
    sum += k[i]*n[i]
    
mu=1.84

p[0]=0.159
nk[0] = N*p[0]
dnk[0] = (nk[0]*(1-(nk[0]/N)))**(1/2)
for i in range(1, len(k)):
    p[i] = mu/k[i]*p[i-1]
    nk[i] = N*p[i]
    
    dnk[i] = (nk[i]*(1-(nk[i]/N)))**(1/2)

for i in range(8):
    #print(i, nk[i]+dnk[i], nk[i], nk[i]-dnk[i])
    pass
    
#f = open("C:\\Users\\angel\\Meine Ablage\\Studium\\Studium Physik\\Praktikum 1\\114v3.txt", "r")
f = open("114v3.txt", "r")
content = f.read()

#print("c: ", content)
f.close()




a = [""]*183
b = [""]*183
k = -1
h = 0
lastd = "\n"
for i in range(len(content)):
    if(content[i]==" "):
        lastd = " "
        k+=1
    elif(content[i]=="\n"):
        lastd = "\n"
        h+=1
    else:
        if(lastd==" "):
            #print(b[h], k, i)
            a[k] += content[i]
        else:
            b[h] += content[i]

N=0
for i in a:
    N+=int(i)
print("N", N)
rez = 10
bars = int(len(a)/rez+1)
c = [0]* bars
for i in range(bars-1):
    for k in range(rez):
        c[i]+=int(a[i*rez+k])

for i in range(len(a)%rez):
    c[bars-1]+=int(a[(bars-1)*rez+i])

#print(a, b)

pg=[0]*bars
e= 2.71828182845904523536028747135266249
pi = 3.1415926535
sum2 = 0
for i in range(bars):
    sum2 += int(c[i])*int(b[i*rez])
    
mu2 = sum2/N
#print("mu2",mu2)
dmu2 = (mu2*N)**(1/2)
for i in range(bars):
    pg[i]=(1/((2*pi*mu2)**(1/2)))*e**(-((int(b[i*rez])-mu2)**2)/(2*mu2))

#print("pg", pg)

s = [0]*bars
for i in range(bars):
    s[i] = 10*N*pg[i]
#print("s", s)

ds = [0]*bars
sk = [0]*bars
dsk = [0]*bars
for i in range(bars):
    #sk[i]=N*pg[i]
    
    dsk[i] = (s[i]*(1-(s[i]/N)))**(1/2)

import math

for i in range(bars):   
    #print("k", int(b[i*rez]), "s", round(c[i], 4), "p", round(pg[i], 4), "sk", round(s[i], 2), "dsk", round(dsk[i], 2))
    pass


#f = open("C:\\Users\\angel\\Meine Ablage\\Studium\\Studium Physik\\Praktikum 1\\114v4.txt", "r")
f = open("114v4.txt", "r")
#f = open("102vb.txt", "r")
contentV4 = f.read()
f.close()
#print(contentV4)
size = 10
rows = 4
x_val = [""]*size
y_val = [""]*size
x_err = [""]*size
y_err = [""]*size

k = 0
h = 0
lastd = "\n"
for i in range(len(contentV4)):
    if(contentV4[i]==" "):
        lastd = " "
        k+=1
    elif(contentV4[i]=="\n"):
        lastd = "\n"
        k+=1
    else:
        if(k%rows==0):
            x_val[int(k/rows)]+=contentV4[i]
            #x_val[int(k/rows)]=(k/rows)+1
            #y_err[int(k/rows)]=0
            #x_err[int(k/rows)]=0
            #continue
        elif(k%rows==1):
            x_err[int(k/rows)]="0"#+=contentV4[i]
        elif(k%rows==2):
            y_val[int(k/rows)]+=contentV4[i]
            #y_err[int(k/rows)]="0"
        elif(k%rows==3):
            y_err[int(k/rows)]+=contentV4[i]

print("yerr", y_err)
print("xval", x_val)
print("yval", y_val)



def convert(x):
    new_x = [0]*len(x)
    for i in range(len(x)):
        new_x[i] = float(x[i])
    return new_x

x_val = convert(x_val)
y_val = convert(y_val)
x_err = convert(x_err)
y_err = convert(y_err)
print(x_val, y_val, x_err, y_err)

#for i in range(len(x_val)):
#    #x_val[i]-=1.4
#    y_val[i]-=1.4
#    y_err[i]=1/y_val[i]*0.1
    
#import math
#for i in range(len(x_val)):
    #x_val[i] = math.log10(x_val[i])
    #y_val[i] = math.log10(y_val[i])
    
print("yerr", y_err)
print("xval", x_val)
print("yval", y_val)

    



def weighted_average(x, xErr):
        
    if(len(x)!=len(xErr)):
        print("Error: x and xErr are not the same length.")
        return
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
        sum1 += x[i]/(xErr[i]**2)
        sum2 += 1/(xErr[i]**2)
        
    return sum1/sum2
def average(x):
    sum = 0
    for i in x:
        sum+=i
    return sum/len(x)


def sq(v):
    x = [0]*len(v)
    for i in range(len(v)):
        x[i] = v[i]**2
    return x
#x_gewichteter_Mittelwert = weighted_average(x_val, y_val)
x_gewichteter_Mittelwert = average(x_val)
y_gewichteter_Mittelwert = weighted_average(y_val, y_err)

xy_val = [0]*len(x_val)
for i in range(len(x_val)):
    xy_val[i]=x_val[i]*y_val[i]
xy_err = [0]*len(x_val)
for i in range(len(x_val)):
    xy_err[i]=x_err[i]*y_err[i]
    
xy_gewichteter_mittelwert = weighted_average(xy_val, y_err)

x_sq_val = sq(x_val)  
y_sq_err = sq(y_err)
#x_sq_gewichteter_mittelwert = weighted_average(x_sq_val, y_err)
x_sq_gewichteter_mittelwert = average(x_sq_val)
m=(xy_gewichteter_mittelwert-x_gewichteter_Mittelwert*y_gewichteter_Mittelwert)/(x_sq_gewichteter_mittelwert-x_gewichteter_Mittelwert**2)
n=(x_sq_gewichteter_mittelwert*y_gewichteter_Mittelwert-x_gewichteter_Mittelwert*xy_gewichteter_mittelwert)/(x_sq_gewichteter_mittelwert-x_gewichteter_Mittelwert**2)

sum=0
err=0
for i in y_err:
    sum+=(1/i**2)
    err=len(y_val)/sum

V_m = 0
for i in range(len(x_val)):
    V_m+=(x_val[i]-x_gewichteter_Mittelwert/(len(x_val)*(x_sq_gewichteter_mittelwert - (x_gewichteter_Mittelwert**2))))**2
V_n = 0
for i in range(len(x_val)):
    V_m+=(x_val[i]-x_gewichteter_Mittelwert*x_val[i]/(len(x_val)*(x_sq_gewichteter_mittelwert - (x_gewichteter_Mittelwert**2))))**2



#print("V_n: ", V_n,"V_m: ", V_m)
import straight_fit

f = straight_fit.fit()

print(f.calc_m_n_Vm_Vn(x_val, x_err, y_val, y_err))
print(m, n, err, x_gewichteter_Mittelwert, y_gewichteter_Mittelwert, xy_gewichteter_mittelwert, x_sq_gewichteter_mittelwert)