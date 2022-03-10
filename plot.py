import sys

import pandas
import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial.transform import Rotation

if len(sys.argv) > 1:
    data = pandas.read_csv(sys.argv[1])
else:
    data = pandas.read_csv(sys.stdin)

rot = Rotation.from_quat(np.array([data.qx,data.qy,data.qz,data.qw]).T)
rot_euler = rot.as_euler('zyx', degrees=True)
euler_df = pandas.DataFrame(data=rot_euler, columns=['yaw', 'pitch', 'roll'])

plt.plot(data.x,-data.z)
plt.xlabel('x-position')
plt.ylabel('z-position (inverted)')
plt.axis('equal')

plt.figure()
plt.plot(data.time,data.x)
plt.plot(data.time,data.z)
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend(['x','z'])

plt.figure()
plt.plot(data.time,euler_df.pitch)
plt.xlabel('Time (s)')
plt.ylabel('Pitch (deg)')

plt.figure()
plt.plot(data.time,np.degrees(data.alpha))
plt.xlabel('Time (s)')
plt.ylabel('Alpha (deg)')

plt.figure()
plt.plot(data.time,data.airspeed)
plt.xlabel('Time (s)')
plt.ylabel('Airspeed (m/s)')

plt.figure()
plt.plot(data.time,data.pitching_moment)
plt.xlabel('Time (s)')
plt.ylabel('Pitching moment (Nm)')

plt.figure()
plt.plot(data.time,data.lift)
plt.xlabel('Time (s)')
plt.ylabel('Lift_Z (N)')

plt.show()