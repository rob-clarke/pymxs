#!/usr/bin/env python3
 
import numpy as np
import utils

import matplotlib.pyplot as plt

from utils import process_dir

import os.path
import pickle

PICKLE_DIR = "../wind_tunnel_data/processed/"
PICKLE_FILE = "data_07_13.5.pkl"

tare1 = utils.tares.create_tare_from_dir('../wind_tunnel_data/raw/2022-04-05/tare1')

# plt.figure()
# plt.scatter(tare_data1.pitch,tare_data1.load_x)
# plt.scatter(tare_data2.pitch,tare_data2.load_x)

# plt.figure()
# plt.scatter(tare_data1.pitch,tare_data1.load_y)
# plt.scatter(tare_data2.pitch,tare_data2.load_y)

# xdata = np.linspace(-15, 15, 200)
# plt.figure()
# plt.scatter(tare_data1.pitch,tare_data1.load_z)
# plt.scatter(xdata,tare1.tarefuncs["load_z"](xdata))
# plt.scatter(tare_data2.pitch,tare_data2.load_z)
# plt.scatter(xdata,tare2.tarefuncs["load_z"](xdata))

pickle_path = os.path.join(PICKLE_DIR,PICKLE_FILE)
if os.path.exists(pickle_path):
    with open(pickle_path,"rb") as f:
        data_13_5_t1 = pickle.load(f)

else:
    data_13_5_t1 = process_dir('../wind_tunnel_data/raw/2022-04-07/as13.5_prop',tare1,has_beta=True,use_april=True)
    data_13_5_t1 = data_13_5_t1.append(process_dir('../wind_tunnel_data/raw/2022-04-08/as13.5_prop',tare1,has_beta=True,use_april=True))

    with open(pickle_path,"wb") as f:
        pickle.dump(data_13_5_t1,f)

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

def select_neutral(data):
    return data[(abs(data.aileron)<3.0) & (abs(data.elevator)<2.0) & (abs(data.rudder)<2.0)]

def qS(airspeed):
    S = 2.625E+05 / 1000**2
    return 0.5 * 1.225 * airspeed**2 * S

sel_13_5p = select_neutral(data_13_5_t1)
#sel_16_5p = sel_16_5p[(abs(sel_16_5p.throttle - 0.8)<0.1) & (abs(sel_16_5p.rig_pitch-5)<1.0)]
# Seems very noisy at -3 beta...

# # Z
# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d: d.rig_pitch,
#     lambda d: d.load_z,
#     "Rig Pitch (deg)",
#     "Load Z (N)"
#     )

# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d,_: d.rig_pitch,
#     lambda d,a: d.load_z / qS(a),
#     "Rig Pitch (deg)",
#     "C_Z",
#     fnargs=[10,15,20]
#     )
    
# # Lift
# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d: d.rig_pitch,
#     lambda d: d.lift,
#     "Rig Pitch (deg)",
#     "Lift (N)"
#     )

# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d,_: d.rig_pitch,
#     lambda d,a: d.lift / qS(a),
#     "Rig Pitch (deg)",
#     "C_L",
#     fnargs=[10,15,20]
#     )

# # Pitching Moment
# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d: d.rig_pitch,
#     lambda d: d.load_m,
#     "Rig Pitch (deg)",
#     "Load M (Nm)"
#     )

# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d,_: d.rig_pitch,
#     lambda d,a: d.load_m / (qS(a) * 0.23),
#     "Rig Pitch (deg)",
#     "C_M",
#     fnargs=[10,15,20]
#     )

# # X
# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d: d.rig_pitch,
#     lambda d: d.load_x,
#     "Rig Pitch (deg)",
#     "Load X (N)"
#     )

# plot_datas(
#     [sel_10,sel_15,sel_20],
#     ('10','15','20'),
#     lambda d,_: d.rig_pitch,
#     lambda d,a: d.load_x / qS(a),
#     "Rig Pitch (deg)",
#     "C_X",
#     fnargs=[10,15,20]
#     )

# Y
plot_datas(
    [sel_13_5p],
    ('13.5'),
    lambda d: d.rig_yaw,
    lambda d: d.load_y,
    "Rig Yaw (deg)",
    "Load Y (N)",
    cfn=lambda d: d.throttle
    )

plot_datas(
    [sel_13_5p],
    ('13.5'),
    lambda d,_: d.rig_yaw,
    lambda d,a: d.load_y / qS(a),
    "Rig Yaw (deg)",
    "C_Y",
    fnargs=[13.5]
    )

# Yaw moment
plot_datas(
    [sel_13_5p],
    ('13.5'),
    lambda d: d.rig_yaw,
    lambda d: d.load_n,
    "Rig Yaw (deg)",
    "Load N (Nm)",
    cfn=lambda d: d.throttle
    )

plot_datas(
    [sel_13_5p],
    ('13.5'),
    lambda d,_: d.rig_yaw,
    lambda d,a: d.load_n / (qS(a) * 0.23),
    "Rig Yaw (deg)",
    "C_N",
    fnargs=[13.5],
    cfn=lambda d,_: d.throttle
    )

plot_datas(
    [sel_13_5p],
    ('13.5'),
    lambda d,_: abs(d.rig_yaw),
    lambda d,a: abs(d.load_n),
    "abs(Rig Yaw) (deg)",
    "abs(Load N) (Nm)",
    fnargs=[13.5],
    cfn=lambda d,_: d.throttle
    )   

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
