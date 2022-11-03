import pickle
import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt

import utils

chord = utils.params.chord

def add_airspeed_column(df,airspeed):
    airspeed = [airspeed]*len(df.index)
    df['airspeed'] = airspeed

sources = [
    ("data_17_10.pkl",10),
    ("data_17_12.5.pkl",12.5),
    ("data_17_15.pkl",15),
    ("data_17_17.5.pkl",17.5),
    ("data_17_20.pkl",20),
    ("data_17_22.5.pkl",22.5),
    ("data_18_10.pkl",10),
    ("data_18_15.pkl",15),
    ("data_18_20.pkl",20)
    ]

data = None
for filename,airspeed in sources:
    newdata = pickle.load(open(filename,"rb"))
    add_airspeed_column(newdata,airspeed)
    data = pd.concat([data,newdata])

def linear(x,m,c):
    return m*x + c

def quadratic(x,a,b,c):
    return a*x**2 + b*x + c

def tanh(x,a,b,c,d):
    return a * np.tanh(b*x + c) + d

# Calculate C_lift
neutral_data = utils.select_neutral(data)

c_lifts = neutral_data.lift / utils.qS(neutral_data.airspeed)

popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_lifts)
C_lift = {
    "zero": float(popt[1]),
    "alpha": float(popt[0])
    }


# Calculate C_drag
c_ds = neutral_data.drag / utils.qS(neutral_data.airspeed)

popt,pcov = scipy.optimize.curve_fit(quadratic,np.radians(neutral_data.pitch),c_ds)
C_drag = {
    "zero": float(popt[2]),
    "alpha": float(popt[1]),
    "alpha2": float(popt[0])
    }

# Calculate C_M
c_ms = neutral_data.load_m / utils.qSc(neutral_data.airspeed)

popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_ms)
C_M = {
    "zero": float(popt[1]),
    "alpha": float(popt[0])
    }

fig,axs = plt.subplots(1,3)

axs[0].scatter(neutral_data.pitch,c_lifts)
fit_samples = np.linspace(-15,15)
fitted = linear(np.radians(fit_samples),C_lift["alpha"],C_lift["zero"])
axs[0].scatter(fit_samples,fitted,c="red")
axs[0].set_xlabel("Pitch angle (deg)")
axs[0].set_ylabel("C_L")

axs[1].scatter(neutral_data.pitch,c_ds)
fit_samples = np.linspace(-15,15)
fitted = quadratic(np.radians(fit_samples),C_drag["alpha2"],C_drag["alpha"],C_drag["zero"])
axs[1].scatter(fit_samples,fitted,c="red")
axs[1].set_xlabel("Pitch angle (deg)")
axs[1].set_ylabel("C_D")

axs[2].scatter(neutral_data.pitch,c_ms)
fit_samples = np.linspace(-15,15)
fitted = linear(np.radians(fit_samples),C_M["alpha"],C_M["zero"])
axs[2].scatter(fit_samples,fitted,c="red")
axs[2].set_xlabel("Pitch angle (deg)")
axs[2].set_ylabel("C_M")

for ax in axs:
    ax.grid(True,'both')

# C_M_elev
elevdata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.rudder)<2.0)
        ]
c_m_alpha = linear(np.radians(elevdata.pitch),C_M["alpha"],C_M["zero"])
c_ms = elevdata.load_m / utils.qSc(elevdata.airspeed) - c_m_alpha
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(elevdata.elevator),c_ms)
C_M["elev_a"] = float(popt[0])
C_M["elev_b"] = float(popt[1])
C_M["elev_c"] = float(popt[2])
C_M["elev_d"] = float(popt[3])


fig,axs = plt.subplots(2,3)

axs[0,0].scatter(elevdata.elevator,c_ms,c=elevdata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_M["elev_a"],C_M["elev_b"],C_M["elev_c"],C_M["elev_d"])
axs[0,0].scatter(fit_samples,fitted,c="red")
axs[0,0].set_xlabel("Elevator angle (deg)")
axs[0,0].set_ylabel("C_M contribution")

# Calculate C_L_ail
aildata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.elevator)<2.0)
        & (abs(data.rudder)<2.0)
        ]

c_ls = aildata.load_l / utils.qSc(aildata.airspeed)
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(aildata.aileron),c_ls)
C_L = {
    "ail_a": float(popt[0]),
    "ail_b": float(popt[1]),
    "ail_c": float(popt[2]),
    "ail_d": float(popt[3])
    }

axs[0,1].scatter(aildata.aileron,c_ls,c=aildata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_L["ail_a"],C_L["ail_b"],C_L["ail_c"],C_L["ail_d"])
axs[0,1].scatter(fit_samples,fitted,c="red")
axs[0,1].set_xlabel("Aileron angle (deg)")
axs[0,1].set_ylabel("C_L contribution")

# Calculate C_L_rudd
rudddata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.elevator)<2.0)
        ]

c_ls = rudddata.load_l / utils.qSc(rudddata.airspeed)
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(rudddata.rudder),c_ls)
C_L["rudd_a"] = float(popt[0])
C_L["rudd_b"] = float(popt[1])
C_L["rudd_c"] = float(popt[2])
C_L["rudd_d"] = float(popt[3])

axs[1,1].scatter(rudddata.rudder,c_ls,c=rudddata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_L["rudd_a"],C_L["rudd_b"],C_L["rudd_c"],C_L["rudd_d"])
axs[1,1].scatter(fit_samples,fitted,c="red")
axs[1,1].set_xlabel("Rudder angle (deg)")
axs[1,1].set_ylabel("C_L contribution")

# Calculate C_N_rudd
rudddata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.elevator)<2.0)
        ]

c_ns = rudddata.load_n / utils.qSc(rudddata.airspeed)
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(rudddata.rudder),c_ns)
C_N = {
    "rudd_a": float(popt[0]),
    "rudd_b": float(popt[1]),
    "rudd_c": float(popt[2]),
    "rudd_d": float(popt[3])
    }

axs[0,2].scatter(rudddata.rudder,c_ns,c=rudddata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_N["rudd_a"],C_N["rudd_b"],C_N["rudd_c"],C_N["rudd_d"])
axs[0,2].scatter(fit_samples,fitted,c="red")
axs[0,2].set_xlabel("Rudder angle (deg)")
axs[0,2].set_ylabel("C_N contribution")

for ax in axs.flatten():
    ax.grid(True,'both')
    

def calculate_c_n_rudd(data):
    c_ns = data.load_n / utils.qSc(data.airspeed)
    popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(data.rudder),c_ns)
    return popt

throttles = [0.2,0.35,0.5,0.65,0.8] 
for thr in throttles:
    thrrudddata = data[
            (abs(data.throttle - thr)<0.05)
            & (abs(data.airspeed - 15)<0.05)
            & (abs(data.rig_pitch - 2.5)<0.5)
            & (abs(data.aileron)<2.0)
            & (abs(data.elevator)<2.0)
        ]
    c_ns = thrrudddata.load_n / utils.qSc(thrrudddata.airspeed)
    axs[1,2].scatter(thrrudddata.rudder,c_ns)
    
    # popt = calculate_c_n_rudd(thrrudddata)
    # fit_samples = np.linspace(-30,30)
    # fitted = tanh(np.radians(fit_samples),*popt)
    # axs[1,2].scatter(fit_samples,fitted)

axs[1,2].legend([str(t) for t in throttles])

for ax in axs.flatten():
    ax.grid(True,'both')

# Calculate C_T
thrdata = data[
        (abs(data.aileron)<2.0)
        & (data.throttle > 0.1)
        & (abs(data.elevator)<2.0)
        & (abs(data.rudder)<2.0)
        ]

c_lift = linear(np.radians(thrdata.pitch),C_lift["alpha"],C_lift["zero"])
c_drag = quadratic(np.radians(thrdata.pitch),C_drag["alpha2"],C_drag["alpha"],C_drag["zero"])
lift = c_lift * utils.qS(thrdata.airspeed)
drag = c_drag * utils.qS(thrdata.airspeed)

# [F_x] = [[ cos(a) -sin(a) ]] [-D]
# [F_z]   [[ sin(a)  cos(a) ]] [-L]
expected_load_x = -drag * np.cos(np.radians(thrdata.pitch)) + lift * np.sin(np.radians(thrdata.pitch))
thrust = thrdata.load_x - expected_load_x

fig, axs = plt.subplots(1,2)
d = axs[0].scatter(thrdata.pitch,thrust,c=thrdata.airspeed)
axs[0].set_xlabel("Pitch angle (deg)")
axs[0].set_ylabel("Thrust (N)")
cbar = fig.colorbar(d,ax=axs[0])
cbar.set_label("Airspeed (m/s)")

rho = 1.225
D = 0.28
n = thrdata.rpm / 60.0

k_t = thrust / (rho * n**2 * D**4)

k_t_select = (thrust > 1.0) & (thrdata.rpm > 100)

d = axs[1].scatter(thrdata[k_t_select].pitch,k_t[k_t_select],c=thrdata[k_t_select].throttle)
axs[1].set_xlabel("Pitch angle (deg)")
axs[1].set_ylabel("Thrust coefficient")
cbar = fig.colorbar(d,ax=axs[1])
cbar.set_label("Throttle setting")

import yaml
print(yaml.dump({
    "C_lift": C_lift,
    "C_drag": C_drag,
    "C_L": C_L,
    "C_M": C_M,
    "C_M": C_N,
    }))

plt.show()