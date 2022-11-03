import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

from matplotlib import rc

rc('font', **{'family': 'serif'})

import matplotlib.pyplot as plt
import numpy as np

from processing_scripts.utils.fits import H, S

samples = np.linspace(-3,3,100)

h_fig = plt.figure()
h_ax = plt.axes()

s_fig = plt.figure()
s_ax = plt.axes()

ks = [2,3,5,10,15]

for k in ks:
    h_ax.plot(samples,H(samples,0,k))
    s_ax.plot(samples,S(samples,-1,1,k))

h_ax.set_xlabel("x")
h_ax.set_ylabel("H'(x,0,k)")
h_ax.legend(list(map(lambda v: f"k = {v}",ks)))
h_ax.grid(True,"both")

s_ax.set_xlabel("x")
s_ax.set_ylabel("S(x,-1,1,k)")
s_ax.legend(list(map(lambda v: f"k = {v}",ks)))
s_ax.grid(True,"both")

plt.show()
