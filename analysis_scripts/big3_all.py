# Do some nonsense to make imports work...
import inspect, os, sys

import pandas
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

from common import load_data, load_beta_data, make_plot

from processing_scripts import utils

from processing_scripts.utils.fits import linear, quadratic, H, S, Fit

import os

import numpy as np
import scipy

import matplotlib.pyplot as plt

def harmonic_fit(xs,ys,mean,harmonics=10, addx=[], addy=[]):
    # https://web.engr.oregonstate.edu/~webbky/MAE4020_5020_files/Section%2010%20Fourier%20Analysis.pdf
    # Let y = C + A_1 cos(wx) + B_1 sin(wx) + A_2 cos(2wx) + B_2 sin(2wx) + ...
    # Z^T Z a = Z^T y; where:
    #  Z = [ 1, cos(wx1), sin(wx1); 1, cos(wx2), sin(wx2); ... ]
    #  a = [ C A_1 B_1 A_2 B_2 ... ]
    
    xs = np.array([*xs, *addx])
    ys = np.array([*ys, *addy])
    
    Zt_list = [ [1] * len(xs) ]
    for i in range(1,harmonics+1):
        Zt_list.append( np.cos(i*xs) )
        Zt_list.append( np.sin(i*xs) )
    
    Zt = np.array(Zt_list)
    
    ZtZ_i = np.linalg.pinv(Zt @ Zt.transpose())
    
    Zty_t = np.empty(
        (1 + harmonics * 2,),
        dtype=xs.dtype
    )
    Zty_t[0] = np.sum(ys)
    Zty_t[1::2] = [ np.sum(ys * np.cos(i*xs)) for i in range(1,harmonics+1) ]
    Zty_t[2::2] = [ np.sum(ys * np.sin(i*xs)) for i in range(1,harmonics+1) ]
    
    Zty = Zty_t.transpose()
    
    return ZtZ_i @ Zty
    
def harmonic_curve(xs,params):
    harmonics = (len(params) - 1) // 2
    return params[0] \
        + np.sum(
            np.array([ params[1+2*i] * np.cos((i+1)*xs) for i in range(harmonics) ]),
            axis=0
        ) \
        + np.sum(
            np.array([ params[2+2*i] * np.sin((i+1)*xs) for i in range(harmonics) ]),
            axis=0
        )


def c_l_curve(alpha,cl_0,cl_alpha,pstall,nstall):
    #astall = np.radians(10)
    return S(alpha,nstall,pstall) * (cl_0 + cl_alpha * alpha) # Linear regime

def calculate_C_lift(neutral_data):
    # Calculate C_lift
    
    c_lifts = neutral_data.lift / utils.qS(neutral_data.airspeed)

    popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_lifts,maxfev=10000)
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
    ax_cl = ax

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
    ax_cd = ax

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
    ax_cm = ax
    return ax_cl, ax_cd, ax_cm

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
    
    ax_cl, ax_cd, ax_cm = make_top3_plots(neutral_data,
        c_lifts,C_lift,C_lift_complex,
        c_ds,C_drag,C_drag_complex,
        c_ms,C_M,C_M_complex
        )
    
    # Load beta data too
    full_data = data.append(load_beta_data(),ignore_index=True)
    neutral_full_data = utils.select_neutral(full_data)
    neutral_full_data = neutral_full_data[(neutral_full_data.rig_yaw == 0) | pandas.isna(neutral_full_data.rig_yaw)]
    # neutral_full_data = neutral_full_data[neutral_full_data.rpm < 500]
    
    
    f_c_lifts,f_C_lift,f_C_lift_complex = calculate_C_lift(neutral_full_data)
    f_c_ds,f_C_drag,f_C_drag_complex = calculate_C_drag(neutral_full_data)
    f_c_ms,f_C_M,f_C_M_complex = calculate_C_M(neutral_full_data)
    
    f_ax_cl, f_ax_cd, f_ax_cm = make_top3_plots(neutral_full_data,
        f_c_lifts,f_C_lift,f_C_lift_complex,
        f_c_ds,f_C_drag,f_C_drag_complex,
        f_c_ms,f_C_M,f_C_M_complex
        )
    
    # plotRange = [-180,180,360]
    # p_h = harmonic_fit(np.radians(neutral_data.pitch), c_lifts, None, 100, addx=np.radians([*np.linspace(-180,-15), *np.linspace(15,180)]),addy=[0]*100)
    # C_harm = harmonic_curve(np.radians(np.linspace(*plotRange)), p_h)
    # ax_cl.plot(np.linspace(*plotRange),C_harm)
    
    import pandas as pd
    xflr_2d_data = pd.read_table(os.path.join(currentdir,'../XFLR/NACA0012Polars/NACA 0012_T1_Re0.250_M0.00_N9.0.txt'),header=0,skiprows=[0,1,2,3,4,5,6,7,8,10],delim_whitespace=True)
    
    alpha_offset = -1.5
    
    ax_cl.plot(xflr_2d_data['alpha']+alpha_offset,xflr_2d_data['CL'],color="pink")
    ax_cd.plot(xflr_2d_data['alpha']+alpha_offset,xflr_2d_data['CD'],color="pink")
    #ax_cm.plot(xflr_2d_data['alpha']+alpha_offset,xflr_2d_data['Cm'],color="pink")
    
    xflr_3d_data = pd.read_table(os.path.join(currentdir,'../XFLR/T1-15_0 m_s-VLM1.txt'),header=0,skiprows=7,delim_whitespace=True)
    ax_cl.plot(xflr_3d_data['alpha']+alpha_offset,xflr_3d_data['CL'],color="purple")
    ax_cd.plot(xflr_3d_data['alpha']+alpha_offset,xflr_3d_data['CD'],color="purple")
    #ax_cm.plot(xflr_3d_data['alpha']+alpha_offset,xflr_3d_data['Cm'],color="purple")
    
    plt.show()
    