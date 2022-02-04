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

class Fit:
    def __init__(self,fn,args):
        self.fn = fn
        self.args = args
        
    def __call__(self,*args):
        return self.fn(*args,*self.args)
