import sys
import json

import pandas
import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial.transform import Rotation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_name", help="Name of run")
parser.add_argument("-d", "--directory", help="Directory for runs", default="./runs")

args = parser.parse_args()

run_dir = f"{args.directory}/{args.run_name}"

data = pandas.read_csv(f"{run_dir}/output.csv")

def get_eulerized(data):
    if "qx" in data:
        rot = Rotation.from_quat(np.array([data.qx,data.qy,data.qz,data.qw]).T)
        rot_euler = rot.as_euler('zyx', degrees=True)
        euler_df = pandas.DataFrame(data=rot_euler, columns=['yaw', 'pitch', 'roll'])
        import copy
        before = copy.deepcopy(euler_df.pitch)
        # The conversion to Euler angles will convert inversions due to pitch into inversions due to roll
        # This mess makes it so only the pitch changes
        is_inverted = euler_df.yaw != 0
        pitch_sign = np.copysign(1,euler_df.pitch)
        euler_df.pitch = (1-is_inverted) * euler_df.pitch + \
            is_inverted * (pitch_sign * 90 + (pitch_sign * 90 - euler_df.pitch))
    else:
        euler_df = data
    return euler_df

data_eul = get_eulerized(data)

with open(f"{run_dir}/metadata.json") as f:
    metadata = json.load(f)

plt.plot(data.x, -data.z)
plt.xlabel('x-position')
plt.ylabel('z-position (inverted)')
plt.axis('equal')
plt.grid('both')

if "waypoints" in metadata:
    plt.scatter(
        list(map(lambda p: p[0],metadata["waypoints"])),
        list(map(lambda p: -p[1],metadata["waypoints"]))
    )

# plt.figure()
fig, ax = plt.subplots(6,1, sharex=True)

ax[-1].set_xlabel('Time (s)')

ax[0].plot(data.time, -data.z)
ax[0].set_ylabel('Height (m)')
ax[0].grid('both')

ax[1].plot(data.time, data_eul.pitch)
ax[1].set_ylabel('Pitch Angle (deg)')
ax[1].grid('both')

ax[2].plot(data.time,np.degrees(data.alpha))
ax[2].set_ylabel('Alpha (deg)')
ax[2].grid('both')

ax[3].plot(data.time, np.hypot(data.u, data.w))
ax[3].set_ylabel('Airspeed (m/s)')
ax[3].grid('both')

ax[4].plot(data.time,np.degrees(data.elevator))
ax[4].set_ylabel('Elevator (deg)')
ax[4].grid('both')

ax[5].plot(data.time,data.throttle)
ax[5].set_ylabel('Throttle (frac)')
ax[5].grid('both')

plt.show()
