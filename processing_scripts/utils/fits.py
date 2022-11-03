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

def poly_manifold(x,y,z,order,*args):    
    if order == 0:
        return args[0]
    if order == 1:
        return args[0] * x + args[1] * y + args[2] * z + args[3]
    if order == 2:
        return args[0]*x*x + args[1]*x*y + args[2]*x*z + args[3]*y*y + args[4]*y*z + args[5]*z*z \
            + args[6]*x + args[7]*y + args[8]*z \
            + args[9]
    if order == 3:
        return args[0]*x*x*x + args[1]*x*x*y + args[2]*x*x*z + args[3]*x*y*y + args[4]*x*y*z + args[5]*x*z*z + args[6]*y*y*x + args[7]*y*y*y + args[8]*y*y*z + args[9]*y*z*z + args[10]*z*z*z \
            + args[11]*x*x + args[12]*x*y + args[13]*y*y + args[14]*x*z + args[15]*y*z + args[16]*z*z \
            + args[17]*x + args[18]*y + args[19]*z \
            + args[20]


class Fit:
    def __init__(self,fn,args):
        self.fn = fn
        self.args = args
        
    def __call__(self,*args):
        return self.fn(*args,*self.args)
