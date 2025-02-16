s = 14*1.66e-27
h = 1.66e-27
vs = (1.6e6*1.6e-19*2/s)**(1/2)
vh = (5.7e6*1.6e-19*2/h)**(1/2)

def p():
    return -s+h-2*(vs*s**2-vh*h**2)/(vs*s-vh*h)

def q():
    return -s*h

def m():
    return (vs*s-vh*h)/(vh-vs)


def me():
    return m()*(3e8)**2/1.6e-19

def v():
    return vs*(s+m())/(2*m())


print(vs, vh)
print(me())
print(m())
print(v())
print(v()**2*m()/2/1.6e-19)
