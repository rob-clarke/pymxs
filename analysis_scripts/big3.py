# Do some nonsense to make imports work...
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

from common import load_data, make_plot

from processing_scripts import utils

from processing_scripts.utils.fits import linear, quadratic, H, S, Fit

import os

import numpy as np
import scipy

import matplotlib.pyplot as plt

def c_l_curve(alpha,cl_0,cl_alpha,pstall,nstall):
    #astall = np.radians(10)
    return S(alpha,nstall,pstall) * (cl_0 + cl_alpha * alpha) # Linear regime

def calculate_C_lift(neutral_data):
    # Calculate C_lift
    
    c_lifts = neutral_data.lift / utils.qS(neutral_data.airspeed)

    popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_lifts)
    C_lift = {
        "zero": float(popt[1]),
        "alpha": float(popt[0])
        }
    popt,pcov = scipy.optimize.curve_fit(c_l_curve,np.radians(neutral_data.pitch),c_lifts,[C_lift["zero"],C_lift["alpha"],np.radians(10),-np.radians(10)])
    C_lift_complex = popt
    
    return c_lifts,C_lift,C_lift_complex


def c_d_curve(alpha,cd_0,cd_alpha,alpha_cd0):
    alpha_lim = 30
    return S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim)) * (cd_alpha*(alpha-alpha_cd0)**2 + cd_0) \
        + (1.0-S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim))) * 2.0

def calculate_C_drag(neutral_data):
    # Calculate C_drag
        
    c_ds = neutral_data.drag / utils.qS(neutral_data.airspeed)

    popt,pcov = scipy.optimize.curve_fit(quadratic,np.radians(neutral_data.pitch),c_ds)
    C_drag = {
        "zero": float(popt[2]),
        "alpha": float(popt[1]),
        "alpha2": float(popt[0])
        }

    popt,pcov = scipy.optimize.curve_fit(c_d_curve,np.radians(neutral_data.pitch),c_ds,maxfev=10000)
    C_drag_complex = popt
    
    return c_ds,C_drag,C_drag_complex


def c_m_curve(alpha,cm_0,alpha_cm0,cm_hscale,cm_vscale):
    alpha_lim = 15
    asymptote = 0.5
    k=12
    return S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim),k) * (cm_vscale*np.tan(cm_hscale*(alpha-alpha_cm0)) + cm_0) \
        + (1.0-H(alpha,np.radians(-alpha_lim),k)) * asymptote \
        + H(alpha,np.radians(alpha_lim),k) * -asymptote

def calculate_C_M(neutral_data):
    # Calculate C_M
        
    c_ms = neutral_data.load_m / utils.qSc(neutral_data.airspeed)

    popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_ms)
    C_M = {
        "zero": float(popt[1]),
        "alpha": float(popt[0])
        }

    popt,pcov = scipy.optimize.curve_fit(c_m_curve,np.radians(neutral_data.pitch),c_ms,maxfev=10000)
    C_M_complex = popt
    
    return c_ms,C_M,C_M_complex

def make_top3_plots(neutral_data,c_lifts,C_lift,C_lift_complex,c_ds,C_drag,C_drag_complex,c_ms,C_M,C_M_complex):
    # Make plots
    #fig,axs = plt.subplots(1,3)

    # Plot lift
    fig = plt.figure()
    ax = plt.axes()
    make_plot(ax,
        neutral_data.pitch,c_lifts,
        np.linspace(-15,15),
        lambda x: linear(np.radians(x),C_lift["alpha"],C_lift["zero"]),
        np.linspace(-30,30,200),
        lambda x: c_l_curve(np.radians(x),*C_lift_complex),
        "Pitch angle (deg)",
        "C_L"
        )

    # Plot drag
    fig = plt.figure()
    ax = plt.axes()
    make_plot(ax,
        neutral_data.pitch,c_ds,
        np.linspace(-15,15),
        lambda x: quadratic(np.radians(x),C_drag["alpha2"],C_drag["alpha"],C_drag["zero"]),
        np.linspace(-30,30,200),
        lambda x: c_d_curve(np.radians(x),*C_drag_complex),
        "Pitch angle (deg)",
        "C_D"
        )

    # Plot C_M
    fig = plt.figure()
    ax = plt.axes()
    make_plot(ax,
        neutral_data.pitch,c_ms,
        np.linspace(-15,15),
        lambda x: linear(np.radians(x),C_M["alpha"],C_M["zero"]),
        np.linspace(-30,30,200),
        lambda x: c_m_curve(np.radians(x),*C_M_complex),
        "Pitch angle (deg)",
        "C_M"
        )

def get_big3_fits():
    data = load_data()
    neutral_data = utils.filters.select_neutral(data)
    
    _,_,C_lift_complex = calculate_C_lift(neutral_data)
    _,_,C_drag_complex = calculate_C_drag(neutral_data)
    _,_,C_M_complex = calculate_C_M(neutral_data)
    
    return Fit(c_l_curve,C_lift_complex), Fit(c_d_curve,C_drag_complex), Fit(c_m_curve,C_M_complex)

if __name__ == "__main__":
    data = load_data()

    neutral_data = utils.select_neutral(data)
    
    c_lifts,C_lift,C_lift_complex = calculate_C_lift(neutral_data)
    c_ds,C_drag,C_drag_complex = calculate_C_drag(neutral_data)
    c_ms,C_M,C_M_complex = calculate_C_M(neutral_data)
    
    make_top3_plots(neutral_data,
        c_lifts,C_lift,C_lift_complex,
        c_ds,C_drag,C_drag_complex,
        c_ms,C_M,C_M_complex
        )
    
    plt.show()
    