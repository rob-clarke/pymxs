import sys

import pandas
import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial.transform import Rotation

have_two = False
data1 = pandas.read_csv(sys.argv[1])
name1 = sys.argv[2]

if len(sys.argv) > 3:
    have_two = True
    data2 = pandas.read_csv(sys.argv[3])
    name2 = sys.argv[4]

def get_eulerized(data):
    if "qx" in data:
        rot = Rotation.from_quat(np.array([data.qx,data.qy,data.qz,data.qw]).T)
        rot_euler = rot.as_euler('zyx', degrees=True)
        euler_df = pandas.DataFrame(data=rot_euler, columns=['yaw', 'pitch', 'roll'])
    else:
        euler_df = data
    return euler_df

data1_eul = get_eulerized(data1)
if have_two:
    data2_eul = get_eulerized(data2)

plt.plot(data1.x,-data1.z)
if have_two:
    plt.plot(data2.x,-data2.z)
plt.xlabel('x-position')
plt.ylabel('z-position (inverted)')
if have_two:
    plt.legend([name1,name2])
plt.axis('equal')
plt.grid('both')

plt.figure()
plt.plot(data1.time,data1.x)
plt.plot(data1.time,data1.z)
if have_two:
    plt.plot(data2.time,data2.x)
    plt.plot(data2.time,data2.z)
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend(
    [f'{name1}: x',f'{name1}: z',f'{name2}: x',f'{name2}: z'] if have_two else [f'{name1}: x',f'{name1}: z']
)
plt.grid('both')

plt.figure()
plt.plot(data1.time,data1_eul.pitch)
if have_two:
    plt.plot(data2.time,data2_eul.pitch)
plt.xlabel('Time (s)')
plt.ylabel('Pitch Angle (deg)')
if have_two:
    plt.legend([name1,name2])
plt.grid('both')

plt.figure()
plt.plot(data1.time,np.degrees(data1.alpha))
if have_two:
    plt.plot(data2.time,np.degrees(data2.alpha))
plt.xlabel('Time (s)')
plt.ylabel('Alpha (deg)')
if have_two:
    plt.legend([name1,name2])
plt.grid('both')

plt.figure()
plt.plot(data1.time,data1.u)
plt.plot(data1.time,data1.v)
plt.plot(data1.time,data1.w)
if have_two:
    plt.plot(data2.time,data2.u)
    plt.plot(data2.time,data2.v)
    plt.plot(data2.time,data2.w)
plt.xlabel('Time (s)')
plt.ylabel('u (m/s)')
if have_two:
    plt.legend([
        f"{name1}_u", f"{name1}_v", f"{name1}_w"
        f"{name2}_u", f"{name2}_v", f"{name2}_w"
        ])
else:
    plt.legend([
        "u", "v", "w"
        ])
plt.grid('both')

plt.figure()
plt.plot(data1.time,np.degrees(data1.elevator))
if have_two:
    plt.plot(data2.time,np.degrees(data2.elevator))
plt.xlabel('Time (s)')
plt.ylabel('Elevator (deg)')
if have_two:
    plt.legend([name1,name2])
plt.grid('both')

# plt.figure()
# plt.plot(data.time,data.pitching_moment)
# plt.xlabel('Time (s)')
# plt.ylabel('Pitching moment (Nm)')

# plt.figure()
# plt.plot(data.time,data.lift)
# plt.xlabel('Time (s)')
# plt.ylabel('Lift_Z (N)')

plt.show()
