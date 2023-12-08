#!/usr/bin/env python3
 
import numpy as np
import utils

import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt

PICKLE_DIR = "../wind_tunnel_data/processed_corrected/"

# tare1 = utils.tares.create_tare_from_dir('../wind_tunnel_data/raw/2022-04-05/tare1')

from utils import process_dir, load_file
from utils.clean import calc_sideforce, transform_data, calc_lift_drag, shift_data
from utils.controls import calc_controls

def load_pickle(pickle_file,loadfn):
    pickle_path = os.path.join(PICKLE_DIR,pickle_file)
    if os.path.exists(pickle_path):
        with open(pickle_path,"rb") as f:
            return pickle.load(f)
    else:
        data = loadfn()
        with open(pickle_path,"wb") as f:
            pickle.dump(data,f)
        return data

def load_data():
    data0 = load_file('../wind_tunnel_data/raw/2021-11-03/','throttle_elev.csv',False)
    data1 = load_file('../wind_tunnel_data/raw/2021-11-04/','all_surfaces.csv',False)

    data = pd.concat([data0, data1])
    data['rig_pitch'] = [0] * len(data.index)

    # tared_data = tare.apply_tare_funcs(raw_data)
    load_columns = [
        'load_x','load_y','load_z','load_l','load_m','load_n'
    ]
    zero_throttle_points = data[data.throttle_pwm < 1100]
    load_means = zero_throttle_points[load_columns].mean()

    tared_data = data.copy()
    for column in load_columns:
        tared_data[column] = data[column] - load_means[column]

    aligned_data = shift_data(transform_data(tared_data))
    augmented_data = calc_lift_drag(calc_controls(aligned_data,use_april=False))

    return augmented_data

augmented_data = load_pickle("data_11_0.pkl", load_data)

import sys
if sys.argv[1] == "noplot":
    sys.exit(0)

# plt.figure()
# plt.scatter(range(len(data_15_t1.index)),data_15_t1.aileron)

# plt.figure()
# plt.scatter(data_10_t1.rig_pitch,data_10_t1.pitch)
# plt.scatter(data_15_t1.rig_pitch,data_15_t1.pitch)
# plt.scatter(data_20_t1.rig_pitch,data_20_t1.pitch)
# plt.legend(('10','15','20'))

def plot_datas(datas,names,xfn,yfn,xlabel="",ylabel="",title="",cfn=None,grid=True,fnargs=None):
    plt.figure()
    if fnargs is None:
        for data in datas:
            plt.scatter(xfn(data),yfn(data),c=(None if cfn is None else cfn(data)) )
    else:
        for data,arg in zip(datas,fnargs):
            plt.scatter(xfn(data,arg),yfn(data,arg),c=(None if cfn is None else cfn(data,arg)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(names)
    if grid:
        plt.grid()

def qS(airspeed):
    S = 2.625E+05 / 1000**2
    return 0.5 * 1.225 * airspeed**2 * S

sel_data = augmented_data

# Z
plot_datas(
    [sel_data],
    ('0'),
    lambda d: d.elevator,
    lambda d: d.load_z,
    "Elevator (deg)",
    "Load Z (N)",
    cfn=lambda d: d.throttle
    )

# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d,_: d.rig_pitch,
#     lambda d,a: d.load_z / qS(a),
#     "Rig Pitch (deg)",
#     "C_Z",
#     fnargs=[10,15,20]
#     )
    
# Lift
plot_datas(
    [sel_data],
    ('0'),
    lambda d: d.elevator,
    lambda d: d.lift,
    "Elevator (deg)",
    "Lift (N)",
    cfn=lambda d: d.throttle
    )

# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d,_: d.rig_pitch,
#     lambda d,a: d.lift / qS(a),
#     "Rig Pitch (deg)",
#     "C_L",
#     fnargs=[5],
#     cfn=lambda d: d.throttle,
#     )

# Pitching Moment
plot_datas(
    [sel_data],
    ('0'),
    lambda d: d.elevator,
    lambda d: d.load_m,
    "Elevator (deg)",
    "Load M (Nm)",
    cfn=lambda d: d.throttle
    )

# elev_samples = np.arange(-40, 40)
# throttle_samples = np.linspace(0, 1, 10)
# fit = lambda e, t: 0.037*t*e + -0.05
# for thr in throttle_samples:
#     plt.plot(elev_samples, fit(elev_samples, thr))

# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d,_: d.elevator,
#     lambda d,a: d.load_m / (qS(a) * 0.23),
#     "Rig Pitch (deg)",
#     "C_M",
#     fnargs=[0.01]
#     )

# X
# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d: d.throttle,
#     lambda d: d.load_x,
#     "Throttle (frac)",
#     "Load X (N)"
#     )

# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d,_: d.throttle,
#     lambda d,a: d.load_x / qS(a),
#     "Throttle (frac)",
#     "C_X",
#     fnargs=[10,15,20]
#     )

# # Y
# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d: d.rig_yaw,
#     lambda d: d.load_y,
#     "Rig Yaw (deg)",
#     "Load Y (N)"
#     )

# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d,_: d.rig_yaw,
#     lambda d,a: d.load_y / qS(a),
#     "Rig Yaw (deg)",
#     "C_Y",
#     fnargs=[10,15,20,10]
#     )

# # Yaw moment
# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d: d.rig_yaw,
#     lambda d: d.load_n,
#     "Rig Yaw (deg)",
#     "Load N (Nm)"
#     )

# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d,_: d.rig_yaw,
#     lambda d,a: d.load_n / (qS(a) * 0.23),
#     "Rig Yaw (deg)",
#     "C_N",
#     fnargs=[10,15,20,10]
#     )

# plot_datas(
#     [sel_data],
#     ('0'),
#     lambda d,_: abs(d.rig_yaw),
#     lambda d,a: abs(d.load_n),
#     "abs(Rig Yaw) (deg)",
#     "abs(Load N) (Nm)",
#     fnargs=[10,15,20,10]
#     )   

# # Drag
# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d: d.rig_pitch,
#     lambda d: d.drag,
#     "Rig Pitch (deg)",
#     "Drag (N)"
#     )

# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d,_: d.rig_pitch,
#     lambda d,a: d.drag / qS(a),
#     "Rig Pitch (deg)",
#     "C_D",
#     fnargs=[10,15,20]
#     )


# # Elevator data
# def select_elev(data):
#     return data[(abs(data.aileron)<3.0) & (abs(data.rudder)<2.0)]

# sel_10 = select_elev(data_10_t1)
# sel_15 = select_elev(data_15_t1)
# sel_20 = select_elev(data_20_t1)

# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d: d.rig_pitch,
#     lambda d: d.load_m,
#     "Rig Pitch (deg)",
#     "Load M (Nm)",
#     cfn=lambda d: d.elevator
#     )

# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d,_: d.rig_pitch,
#     lambda d,a: d.load_m / (qS(a) * 0.23),
#     "Rig Pitch (deg)",
#     "C_M",
#     cfn=lambda d,_: d.elevator,
#     fnargs=[10,15,20]
#     )

plt.show()
