#!/usr/bin/env python3

import os
import pickle
 
import numpy as np
import utils

import matplotlib.pyplot as plt

from utils import process_dir, select_neutral, select_elev, plot_datas, qS

tare1 = utils.tares.create_tare_from_dir('../wind_tunnel_data/raw/2021-11-17/tare')

PICKLE_DIR = "../wind_tunnel_data/processed_corrected/"

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

data_10 = load_pickle("data_17_10.pkl", lambda: process_dir('../wind_tunnel_data/raw/2021-11-17/as10',tare1))
data_12 = load_pickle("data_17_12.5.pkl", lambda: process_dir('../wind_tunnel_data/raw/2021-11-17/as12.5',tare1))
data_15 = load_pickle("data_17_15.pkl", lambda: process_dir('../wind_tunnel_data/raw/2021-11-17/as15',tare1))
data_17 = load_pickle("data_17_17.5.pkl", lambda: process_dir('../wind_tunnel_data/raw/2021-11-17/as17.5',tare1))
data_20 = load_pickle("data_17_20.pkl", lambda: process_dir('../wind_tunnel_data/raw/2021-11-17/as20',tare1))
data_22 = load_pickle("data_17_22.5.pkl", lambda: process_dir('../wind_tunnel_data/raw/2021-11-17/as22.5',tare1))

import sys
if sys.argv[1] == "noplot":
    sys.exit(0)

datas = [
    data_10,
    data_12,
    data_15,
    data_17,
    data_20,
    data_22,
    ]

speeds = [10,12.5,15,17.5,20,22.5]
names = ('10','12.5','15','17.5','20','22.5')

sel_datas = list(map(lambda d: select_neutral(d),datas))

# Z
plot_datas(
    sel_datas,names,
    lambda d: d.rig_pitch,
    lambda d: d.load_z,
    "Rig Pitch (deg)",
    "Load Z (N)"
    )

plot_datas(
    sel_datas,names,
    lambda d,_: d.rig_pitch,
    lambda d,a: d.load_z / qS(a),
    "Rig Pitch (deg)",
    "C_Z",
    fnargs=speeds
    )
    
# Lift
plot_datas(
    sel_datas,names,
    lambda d: d.rig_pitch,
    lambda d: d.lift,
    "Rig Pitch (deg)",
    "Lift (N)"
    )

plot_datas(
    sel_datas,names,
    lambda d,_: d.rig_pitch,
    lambda d,a: d.lift / qS(a),
    "Rig Pitch (deg)",
    "C_L",
    fnargs=speeds
    )

# Pitching Moment
plot_datas(
    sel_datas,names,
    lambda d: d.rig_pitch,
    lambda d: d.load_m,
    "Rig Pitch (deg)",
    "Load M (Nm)"
    )

plot_datas(
    sel_datas,names,
    lambda d,_: d.rig_pitch,
    lambda d,a: d.load_m / (qS(a) * 0.23),
    "Rig Pitch (deg)",
    "C_M",
    fnargs=speeds
    )

# X
plot_datas(
    sel_datas,names,
    lambda d: d.rig_pitch,
    lambda d: d.load_x,
    "Rig Pitch (deg)",
    "Load X (N)"
    )

plot_datas(
    sel_datas,names,
    lambda d,_: d.rig_pitch,
    lambda d,a: d.load_x / qS(a),
    "Rig Pitch (deg)",
    "C_X",
    fnargs=speeds
    )

# Drag
plot_datas(
    sel_datas,names,
    lambda d: d.rig_pitch,
    lambda d: d.drag,
    "Rig Pitch (deg)",
    "Drag (N)"
    )

plot_datas(
    sel_datas,names,
    lambda d,_: d.rig_pitch,
    lambda d,a: d.drag / qS(a),
    "Rig Pitch (deg)",
    "C_D",
    fnargs=speeds
    )


sel_datas = list(map(lambda d: select_elev(d),datas))

plot_datas(
    sel_datas,names,
    lambda d: d.rig_pitch,
    lambda d: d.load_m,
    "Rig Pitch (deg)",
    "Load M (Nm)",
    cfn=lambda d: d.elevator
    )

plot_datas(
    sel_datas,names,
    lambda d,_: d.rig_pitch,
    lambda d,a: d.load_m / (qS(a) * 0.23),
    "Rig Pitch (deg)",
    "C_M",
    cfn=lambda d,_: d.elevator,
    fnargs=speeds
    )

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for d in sel_datas:
    ax.scatter(d.rig_pitch,d.elevator,d.load_m,c=d.throttle)

ax.set_xlabel("Rig pitch (deg)")
ax.set_ylabel("Elevator (deg)")
ax.set_zlabel("Load M (Nm)")

plt.show()
