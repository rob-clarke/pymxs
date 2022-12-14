import sys

import pandas
import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial.transform import Rotation

data = pandas.read_csv(sys.argv[1])

def get_eulerized(data):
    if "qx" in data:
        rot = Rotation.from_quat(np.array([data.qx,data.qy,data.qz,data.qw]).T)
        rot_euler = rot.as_euler('zyx', degrees=True)
        euler_df = pandas.DataFrame(data=rot_euler, columns=['yaw', 'pitch', 'roll'])
    else:
        euler_df = data
    return euler_df

data_eul = get_eulerized(data)

plt.plot(data.x, -data.z)
plt.xlabel('x-position')
plt.ylabel('z-position (inverted)')
plt.axis('equal')
plt.grid('both')

plt.figure()
fig, ax = plt.subplots(5,1, sharex=True)

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

plt.show()
