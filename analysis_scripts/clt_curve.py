#!/usr/bin/env python

# Do some nonsense to make imports work...
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

from common import load_data, make_plot

from processing_scripts import utils

from processing_scripts.utils.fits import linear, quadratic, H, S, Fit

import matplotlib.pyplot as plt
import numpy as np
import math

def cl_t(alpha):
	cl_t_alpha = 0.5/np.radians(8)
	cl_t_alpha_lin_max = np.radians(10)
	k=10
	# print("c_lt_alpha",cl_t_alpha)
	# print("cl_t_alpha_lin_max",cl_t_alpha_lin_max)
	return cl_t_alpha * alpha * S(alpha,-cl_t_alpha_lin_max,cl_t_alpha_lin_max,k) \
		+ 0.65 * H(alpha,cl_t_alpha_lin_max,k) \
		- 0.65 * (1-H(alpha,-cl_t_alpha_lin_max,k))

samples = np.radians(np.linspace(-30,30))

y = cl_t(samples)

plt.plot(np.degrees(samples),y)
plt.minorticks_on()
plt.grid(visible=True,which='both')

alpha = np.radians(5)
x_t = -0.585
v_inf = 15

def alpha_t(q,alpha=alpha,v_inf=v_inf):
	return np.arctan((-q * x_t + v_inf*np.sin(alpha)) / (v_inf*np.cos(alpha)))

def damp_moment(q,alpha=alpha,v_inf=v_inf):
	alpha_t_val = alpha_t(q,alpha,v_inf)
	M = (x_t - -0.2) * 1.225 * 0.5 * v_inf**2 * ( cl_t(alpha_t_val) - cl_t(alpha) )
	return M

plt.figure()
samples = np.radians(np.linspace(-300,300))
plt.plot(np.degrees(samples),damp_moment(samples))

plt.show()
