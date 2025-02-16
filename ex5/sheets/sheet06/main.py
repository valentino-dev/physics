import numpy as np


a = np.array([15.67, 17.23, 0.714, 93.15, 11.2])*1e6


delta = lambda Aa, Zz: 1 if (Aa%2==1 & Zz%2==1) else (-1 if (Aa%2==0 & Zz%2==0) else 0)

M_n = 939.59e6
M_p = 938.27e6
M_e = 0.511e6
def M(A, Z):
    return (A-Z)*M_n+Z*M_p+Z*M_e-a[0]*A+a[1]*A**(2/3)+a[2]*Z**2/A**(1/3)+a[3]*(A-2*Z)**2/4/A+a[4]*np.array([delta(Aa, Zz) for Aa, Zz in zip(A, Z)]) / A**(1/2)

def C(A, Z):
    return a[2]*Z**2/A**(1/3)



A = np.array([2, 4, 6, 56])
Z = np.array([1, 2, 3, 26])

def B(A, Z):
    #return Z*M(np.array([1]), np.array([1]))+(A-Z)*M_n-M(A,Z)
    return Z*M_p+(A-Z)*M_n-M(A,Z)

#print(M(A,Z)*1e-6)


mass = np.array([1876.14, 3728.39, 5603.05, 52102.1])*1e6

energy = Z*M_p+(A-Z)*M_n-mass

#print(energy*1e-6)
#print(energy*1e-6/A)
#print(energy*1e-6/A)

isobar = 52
#Z = np.arange(start=A[0]-area/2, stop=A[0]+area/2+1, dtype=np.int32)
Z = np.arange(start=0, stop=isobar, dtype=np.int32)
A = np.zeros(Z.shape[0], dtype=np.int32)+isobar
#print(A.shape, Z.shape)
#print(A, Z)
masses = M(A, Z)
#print(masses)

#np.savetxt("3_1.dat", np.concatenate(([masses*1e-9], [Z])).T , delimiter=" ")


def Z_min(A):
    A = int(A)
    Z = np.arange(start=0, stop=A+11, dtype=np.int32)
    A = np.zeros(Z.shape[0], dtype=np.int32)+A
    Z_min = Z[np.argmin(M(A, Z))]
    return Z_min

As = np.arange(300)
Z_mins = np.array([Z_min(A) for A in As])
np.savetxt("1_2.dat", np.concatenate(([Z_mins], [As-Z_mins])).T , delimiter=" ")




A = np.array([94, 92])
Z = np.array([239, 235])
#print(M(A, Z)*1e-6)



a_v=15.67
a_s=17.23
a_c=0.714
a_a=93.15
E_B=28.3
def q(x, y):
    return -4*a_v+8/3*a_s*x**(-1/3)+4*a_c*y*(1-y/3/x)*x**(-1/3)-a_a*(x-2*y)**2/x**2+E_B


print(q(107, 47))
print(q(197, 79))
print(q(238, 92))

qs = np.array([q(As[i], Z_mins[i]) for i in range(As.shape[0])])
np.savetxt('1_2_q.dat', np.concatenate(([As], [qs])).T, delimiter=" ")


print((B(np.array([236]), np.array([92]))-2*B(np.array([168]), np.array([46])))*1e-6)
print((C(np.array([236]), np.array([92]))-2*C(np.array([168]), np.array([46])))*1e-6)

