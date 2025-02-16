import numpy as np


a = np.array([15.67, 17.23, 0.714, 93.15, 11.2])*1e6


delta = lambda Aa, Zz: 1 if (Aa%2==1 & Zz%2==1) else (-1 if (Aa%2==0 & Zz%2==0) else 0)

M_n = 939.59e6
M_p = 938.27e6
M_e = 0.511e6
def M(A, Z):
    return (A-Z)*M_n+Z*M_p+Z*M_e-a[0]*A+a[1]*A**(2/3)+a[2]*Z**2/A**(1/3)+a[3]*(A-2*Z)**2/4/A+a[4]*np.array([delta(Aa, Zz) for Aa, Zz in zip(A, Z)]) / A**(1/2)

A = np.array([2, 4, 6, 56])
Z = np.array([1, 2, 3, 26])

def B(A, Z):
    #return Z*M(np.array([1]), np.array([1]))+(A-Z)*M_n-M(A,Z)
    return Z*M_p+(A-Z)*M_n-M(A,Z)

print(M(A,Z)*1e-6)


mass = np.array([1876.14, 3728.39, 5603.05, 52102.1])*1e6

energy = Z*M_p+(A-Z)*M_n-mass

print(energy*1e-6)
print(energy*1e-6/A)
print(energy*1e-6/A)

isobar = 52
#Z = np.arange(start=A[0]-area/2, stop=A[0]+area/2+1, dtype=np.int32)
Z = np.arange(start=0, stop=isobar, dtype=np.int32)
A = np.zeros(Z.shape[0], dtype=np.int32)+isobar
print(A.shape, Z.shape)
print(A, Z)
masses = M(A, Z)
print(masses)

np.savetxt("3_1.dat", np.concatenate(([masses*1e-9], [Z])).T , delimiter=" ")


def Z_min(A):
    A = int(A)
    Z = np.arange(start=0, stop=A+11, dtype=np.int32)
    A = np.zeros(Z.shape[0], dtype=np.int32)+A
    Z_min = Z[np.argmin(M(A, Z))]
    return Z_min

As = np.arange(100)
Z_mins = np.array([Z_min(A) for A in As])
np.savetxt("3_3.dat", np.concatenate(([Z_mins], [As-Z_mins])).T , delimiter=" ")


A = np.array([94, 92])
Z = np.array([239, 235])
print(M(A, Z)*1e-6)




