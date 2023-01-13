# Do some nonsense to make imports work...
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

from matplotlib import rc

rc('font', **{'family': 'serif'})

from common import load_data, make_plot

from processing_scripts import utils

from processing_scripts.utils.fits import linear, quadratic, H, S, Fit

import os
import math

import numpy as np
import scipy

import matplotlib.pyplot as plt

def c_l_curve(alpha,cl_0,cl_alpha,pstall,nstall):
    k_stall = 25
    k_decay = 10
    a_pdecay = math.radians(30)
    a_ndecay = math.radians(-37)
    linear_regime = S(alpha,nstall,pstall,k_stall) * (cl_0 + cl_alpha * alpha)

    post_stall_pos = H(alpha,pstall,k_stall) * (1-H(alpha,a_pdecay,k_decay)) * (alpha + 0.45)
    high_alpha_regime = H(alpha,a_pdecay,k_decay) * 0.8 * ((math.pi/2)-alpha) * (1-H(alpha,math.radians(160),7))

    post_stall_neg = H(alpha,a_ndecay,k_decay) * (1-H(alpha,nstall,k_stall)) * (alpha - 0.35)
    low_alpha_regime = (1-H(alpha,a_ndecay,k_decay)) * 0.8 * ((-math.pi/2)-alpha) * H(alpha,math.radians(-160),7)

    return linear_regime \
        + post_stall_pos + high_alpha_regime \
        + post_stall_neg + low_alpha_regime

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
    alpha_lim = math.radians(40)
    k = 2
    return S(alpha,-alpha_lim,alpha_lim,k) * (cd_alpha*(alpha-alpha_cd0)**2 + cd_0) \
        + (1.0-S(alpha,-alpha_lim,alpha_lim,k)) * 1.8

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
    alpha_lim = math.radians(15)
    nalpha_lim = math.radians(-15)
    asymptote = 0.802
    k=12
    return S(alpha,nalpha_lim,alpha_lim,k) * (cm_vscale*np.tan(cm_hscale*(alpha-alpha_cm0)) + cm_0) \
        + (1.0-H(alpha,nalpha_lim,k)) * asymptote \
        + H(alpha,alpha_lim,k) * -asymptote

def calculate_C_M(neutral_data):
    # Calculate C_M

    c_ms = neutral_data.load_m / utils.qSc(neutral_data.airspeed)

    popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_ms)
    C_M = {
        "zero": float(popt[1]),
        "alpha": float(popt[0])
        }

    popt,pcov = scipy.optimize.curve_fit(c_m_curve,np.radians(neutral_data.pitch),c_ms,maxfev=100000)
    C_M_complex = popt

    return c_ms,C_M,C_M_complex

def c_lta(alpha):
    a_stall = math.radians(10)
    a_decay = math.radians(40)
    return 3.5810*S(alpha,-a_stall,a_stall)*alpha \
        + 0.65 * S(alpha,a_stall,a_decay) \
        + (math.pi/2-0.25-0.8*alpha) * S(alpha,a_decay,math.pi) * (1-H(alpha,math.radians(145),3)) \
        - 0.65 * S(alpha,-a_decay,-a_stall) \
        + (-math.pi/2+0.25-0.8*alpha) * S(alpha,-math.pi,-a_decay) * H(alpha,math.radians(-145),3)

def c_dta(alpha):
    a_lim = np.radians(9)
    ao_lim = np.radians(30)
    am_lim = np.radians(150)
    return 2 * S(alpha,-a_lim,a_lim,12) * (0.5*alpha)**2 \
        + S(alpha,-ao_lim,ao_lim,10) * (1-S(alpha,-a_lim,a_lim,12)) * 1.5 * (np.abs(alpha) - 0.05) \
        + 1.05 * (1-S(alpha,-ao_lim,ao_lim,10)) * S(alpha,-am_lim,am_lim,5) \
        + 0.018

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
        None, #np.linspace(-30,30,200),
        None, #lambda x: c_l_curve(np.radians(x),*C_lift_complex),
        "Pitch angle (deg)",
        "Lift coefficient"
        )
    ax_cl = ax

    # Plot drag
    fig = plt.figure()
    ax = plt.axes()
    make_plot(ax,
        neutral_data.pitch,c_ds,
        np.linspace(-15,15),
        lambda x: quadratic(np.radians(x),C_drag["alpha2"],C_drag["alpha"],C_drag["zero"]),
        None, #np.linspace(-30,30,200),
        None, #lambda x: c_d_curve(np.radians(x),*C_drag_complex),
        "Pitch angle (deg)",
        "Drag coefficient"
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
        "Pitch moment coefficient"
        )
    ax_cm = ax

    return ax_cl, ax_cd, ax_cm


S_w = 0.263
c_w = 0.24
S_t = 0.0825
c_t = 0.165
x_t = -0.585 - -0.09 - (c_t*0.25)

q = 0.5 * 1.225 * 15**2
tail_setting_angle = math.radians(-0.75) # +ve = upward from reference

def tail_moment(alpha):
    lift = q * S_t * c_lta(alpha)
    drag = q * S_t * c_dta(alpha)
    resolved = lift * np.cos(alpha) + drag * np.sin(alpha)
    return x_t * resolved



def get_big3_fits():
    data = load_data()
    neutral_data = utils.filters.select_neutral(data)

    _,_,C_lift_complex = calculate_C_lift(neutral_data)
    _,_,C_drag_complex = calculate_C_drag(neutral_data)
    _,_,C_M_complex = calculate_C_M(neutral_data)

    def downwash_angle(alpha, derate=True):
        c_lw = c_l_curve(alpha,*C_lift_complex)
        return 2 * c_lw / (np.pi * 4.54) * ( S(alpha,-math.pi/4,math.pi/4) if derate else 1.0 )

    def effective_cm(alpha,derate=True):
        alpha = alpha - downwash_angle(alpha,derate) + tail_setting_angle
        return tail_moment(alpha) / (q*S_w*c_w)

    # return Fit(c_l_curve,C_lift_complex), Fit(c_d_curve,C_drag_complex), Fit(c_m_curve,C_M_complex)
    return Fit(c_l_curve,C_lift_complex), Fit(c_d_curve,C_drag_complex), Fit(effective_cm, [True])

if __name__ == "__main__":
    import sys
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

    print(f"C_lift_complex = {repr(list(C_lift_complex))}")
    print(f"C_drag_complex = {repr(list(C_drag_complex))}")
    print(f"C_M_complex = {repr(list(C_M_complex))}")

    if len(sys.argv) > 1 and sys.argv[1] == "noplot":
        sys.exit(0)

    import pandas as pd
    xflr_2d_data = pd.read_table(
        os.path.join(currentdir,'../xflr_data/NACA0012Polars/NACA 0012_T1_Re0.250_M0.00_N9.0.txt'),
        header=0,
        skiprows=[0,1,2,3,4,5,6,7,8,10],
        delim_whitespace=True
    )

    alpha_offset = -1.5

    ax_cl.plot(xflr_2d_data['alpha']+alpha_offset,xflr_2d_data['CL'],color="pink")
    ax_cd.plot(xflr_2d_data['alpha']+alpha_offset,xflr_2d_data['CD'],color="pink")
    ax_cm.plot(xflr_2d_data['alpha']+alpha_offset,xflr_2d_data['Cm'],color="pink")

    xflr_3d_data = pd.read_table(
        os.path.join(currentdir,'../xflr_data/T1-15_0 m_s-VLM1.txt'),
        header=0,
        skiprows=7,
        delim_whitespace=True
    )
    ax_cl.plot(xflr_3d_data['alpha']+alpha_offset,xflr_3d_data['CL'],color="purple")
    ax_cd.plot(xflr_3d_data['alpha']+alpha_offset,xflr_3d_data['CD'],color="purple")
    ax_cm.plot(xflr_3d_data['alpha']+alpha_offset,xflr_3d_data['Cm'],color="purple")

    def downwash_angle(alpha, derate=True):
        c_lw = c_l_curve(alpha,*C_lift_complex)
        return 2 * c_lw / (np.pi * 4.54) * ( S(alpha,-math.pi/4,math.pi/4) if derate else 1.0 )

    def effective_cm(alpha,derate=True):
        alpha = alpha - downwash_angle(alpha,derate) + tail_setting_angle
        return tail_moment(alpha) / (q*S_w*c_w)

    alpha_samples = np.linspace(-180,180,500)

    ax_cl.plot(alpha_samples,c_l_curve(np.radians(alpha_samples),*C_lift_complex))
    ax_cd.plot(alpha_samples,c_d_curve(np.radians(alpha_samples),*C_drag_complex))

    ax_cl.set_xlim([-35,35])
    ax_cl.set_ylim([-1.3,1.3])
    ax_cl.legend(['Linear fit','XFLR 2D','XFLR 3D','C_L fit'])

    ax_cd.set_xlim([-35,35])
    ax_cd.set_ylim([0,1.0])
    ax_cd.legend(['Quadratic fit','XFLR 2D','XFLR 3D','C_D fit'])

    ax_cm.plot(alpha_samples,effective_cm(np.radians(alpha_samples),False))
    ax_cm.plot(alpha_samples,effective_cm(np.radians(alpha_samples),True))

    # ax_cm.plot(
    #     alpha_samples,
    #     c_m_curve(
    #         np.radians(alpha_samples),
    #         0.053,
    #         0.0484,
    #         1.4151,
    #         -0.5462
    #     )
    # )

    def c_m_curve_noast(alpha,cm_0,alpha_cm0,cm_hscale,cm_vscale):
        alpha_lim = 17
        asymptote = 0.802
        k=10
        return (cm_vscale*np.tan(cm_hscale*(alpha-alpha_cm0)) + cm_0)

    # ax_cm.plot(
    #     alpha_samples,
    #     c_m_curve_noast(
    #         np.radians(alpha_samples),
    #         0.053,
    #         0.0484,
    #         1.4151,
    #         -0.5462
    #     )
    # )

    # ax_cm.plot(
    #     alpha_samples,
    #     c_m_curve_noast(
    #         np.radians(alpha_samples),
    #         *C_M_complex
    #     )
    # )
    ax_cm.set_xlim([-45,45])
    ax_cm.set_ylim([-1,1])
    ax_cm.legend(['Linear fit','Asymptopic fit','XFLR 2D','XFLR 3D','Tail-derived coefficient','TDC downwash fade'])

    plt.figure()
    plt.plot(alpha_samples,c_dta(np.radians(alpha_samples)))
    plt.grid(True)

    plt.figure()
    plt.plot(alpha_samples,c_lta(np.radians(alpha_samples)))
    plt.grid(True)

    plt.figure()
    plt.plot(alpha_samples,tail_moment(np.radians(alpha_samples)))
    plt.grid(True)
    plt.ylabel("Tail moment")
    plt.xlabel("Alpha (deg)")

    plt.show()
