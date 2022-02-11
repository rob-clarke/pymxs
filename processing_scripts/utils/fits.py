import numpy as np

def linear(x,m,c):
    return m*x + c

def quadratic(x,a,b,c):
    return a*x**2 + b*x + c

def tanh(x,a,b,c,d):
    return a * np.tanh(b*x + c) + d

def H(x,p,k=10):
    # Approx heaviside: Return ~0 for all x < p & 1 for all x > p
    # As is approx, is differentiable but there is a rolloff between regimes
    return 1/(1+np.exp(-2*k*(x-p)))

def S(x,l,h,k=10):
    # Approx heavicentre: Return ~1 for all l < x < h, else 0.0
    # As is approx, there is rolloff
    return H(x,l,k) * (1-H(x,h,k))

def P(x,*args):
    result = 0
    for i,coef in enumerate(args):
        result += coef*pow(x,i)
    return result

def poly_surface(x,y,order,*args):
    # 0 -> c
    #   1  (0,0)
    # 1 -> x y c
    #   3  (1,0) (0,1) (0,0)
    # 2 -> xx xy yy x y c
    #   6  (2,0) (1,1) (0,2) (1,0) (0,1) (0,0)
    # 3 -> xxx xxy xyy yyy xx xy yy x y c
    #   10
    # 4 -> xxxx xxxy xxyy xyyy yyyy xxx xxy xyy yyy xx xy yy x y c
    #   15 <- Triangle numbers!
    argindex = 0
    result = 0.0
    for i in range(order+1)[::-1]:
        xpow = i
        ypow = 0
        while xpow >= 0:
            result += args[argindex] * np.power(x,xpow) * np.power(y,ypow)
            ypow += 1
            xpow -= 1
            argindex += 1
    return result

class Fit:
    def __init__(self,fn,args):
        self.fn = fn
        self.args = args
        
    def __call__(self,*args):
        return self.fn(*args,*self.args)
