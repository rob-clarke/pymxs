import sys
import json

import pandas
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import numpy as np

from scipy.spatial.transform import Rotation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_name", help="Name of run")
parser.add_argument("-d", "--directory", help="Directory for runs", default="./runs")

args = parser.parse_args()

run_dir = f"{args.directory}/{args.run_name}"

data = pandas.read_csv(f"{run_dir}/output.csv")

PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"

def parse_path(path):
    vertices = []
    codes = []
    parts = path.split()
    code_map = {
        'M': Path.MOVETO,
        'C': Path.CURVE4,
        'L': Path.LINETO,
    }
    i = 0
    while i < len(parts) - 1:
        if parts[i] in code_map:
            path_code = code_map[parts[i]]
            code_len = 1
        else:
            path_code = code_map['l']
            code_len = 0
        npoints = Path.NUM_VERTICES_FOR_CODE[path_code]
        codes.extend([path_code] * npoints)
        vertices.extend([[*map(float, y.split(','))]
                         for y in parts[i+code_len:][:npoints]])
        i += npoints + code_len
    return vertices, codes

def draw_vehicle(ax, x, y, pitch, elev, scale = 0.1):
    transform = np.array([
        [np.cos(pitch), -np.sin(pitch)],
        [np.sin(pitch),  np.cos(pitch)]
    ])
    for path in [PLANE_OUTLINE_PATH, MAIN_WING_PATH, TAIL_PLANE_PATH]:
        vertices, codes = parse_path(path)
        vertices = scale * (transform @ np.array(vertices).T).T
        positioned = vertices + np.array([x,y])
        ax.add_patch(PathPatch(Path(positioned, codes=codes, closed=True), fill=False))

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

ax = plt.gca()
last_pos = None
for i in range(len(data.index)):
    if i % 10 != 0:
        continue
    if last_pos is None:
        last_pos = [data.x[i], data.z[i]]
    deltaPos = np.hypot(last_pos[0] - data.x[i], last_pos[1] - data.z[i])
    # if deltaPos < 0.5:
    #     continue
    draw_vehicle(ax, data.x[i], -data.z[i], np.radians(data_eul.pitch[i]), 0)
    last_pos = [data.x[i], data.z[i]]

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
