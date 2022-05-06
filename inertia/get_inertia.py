#!/usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation
import scipy.optimize
import pandas as pd
import plotly.express as px

def load_file(path):
    with open(path,'r') as f:
        lines = f.readlines()
        # Line 2 [1] Contains system frequency
        # Line 3 [2] Contains object names
        # Line 4 [3] Contains column names
        # Line 5 [4] Contains units

        system_freq = float(lines[1])

        object_names_raw = lines[2].split(',')
        object_names = []
        for i,name_raw in enumerate(object_names_raw):
            object_name = name_raw.strip().split(':')[-1]
            if object_name:
                object_names.append((i,object_name))

        column_names = lines[3].split(',')
        for i,column_name in enumerate(column_names):
            object_name = ""
            for index,object_name_candidate in object_names:
                if i < index:
                    break
                object_name = object_name_candidate
            if object_name != "":
                column_name = "_".join([object_name,column_name.strip()])
            column_names[i] = column_name

        #units = lines[4].split(',')

        array = np.zeros(shape=(len(lines)-5,len(column_names)))

        for i,line in enumerate(lines[5:]):
            array[i,:] = np.array(
                list(map(
                    lambda v: float(v) if len(v) else np.NAN,
                    line.strip().split(',')
                ))
            )

        return pd.DataFrame(array,columns=column_names)


def filter_columns(df):
    def should_keep_column(name):
        return \
            "MXS" in name \
            or "LowerTriangle" in name \
            or "Frame" in name
    
    drop_names = filter(lambda v: not should_keep_column(v),df.columns)
    return df.drop(columns=list(drop_names))

def filter_nans(df):
    return df.dropna(subset=["MXS_RX","MXS_RY","MXS_RZ"])

def get_rotations(data):
    return Rotation.from_rotvec(np.array([data.MXS_RX,data.MXS_RY,data.MXS_RZ]).transpose())

def get_triangle_rotations(data):
    return Rotation.from_rotvec(np.array([data.LowerTriangle_RX,data.LowerTriangle_RY,data.LowerTriangle_RZ]).transpose())

triangledata = filter_columns(load_file("Data/MXSInertia_Triangle.csv")).dropna(subset=["LowerTriangle_RX","LowerTriangle_RY","LowerTriangle_RZ"])

triangledata["experiment"] = "triangle"

blocksdata = filter_columns(load_file("Data/MXSInertia_RollBlocks.csv")).dropna(subset=["LowerTriangle_RX","LowerTriangle_RY","LowerTriangle_RZ"])

blocksdata["experiment"] = "blocks"

yawdata = filter_nans(filter_columns(load_file("Data/MXSInertia_Yaw.csv")))
pitchdata = filter_nans(filter_columns(load_file("Data/MXSInertia_Pitch.csv")))
rolldata = filter_nans(filter_columns(load_file("Data/MXSInertia_Roll.csv")))

yawdata["experiment"] = "yaw"
pitchdata["experiment"] = "pitch"
rolldata["experiment"] = "roll"

yawbattdata = filter_nans(filter_columns(load_file("Data/MXSInertia_YawBattery.csv")))
pitchbattdata = filter_nans(filter_columns(load_file("Data/MXSInertia_PitchBattery.csv")))
rollbattdata = filter_nans(filter_columns(load_file("Data/MXSInertia_RollBattery.csv")))

yawbattdata["experiment"] = "yawbatt"
pitchbattdata["experiment"] = "pitchbatt"
rollbattdata["experiment"] = "rollbatt"

data = pd.concat([triangledata,blocksdata,yawdata,pitchdata,rolldata,yawbattdata,pitchbattdata,rollbattdata])

rotations = get_rotations(data).as_euler('ZYX',degrees=True)

data['mxs_yaw'] = rotations[:,0]
data['mxs_pitch'] = rotations[:,1]
data['mxs_roll'] = rotations[:,2]

trirotations = get_triangle_rotations(data).as_euler('ZYX',degrees=True)

data['tri_yaw'] = trirotations[:,0]
data['tri_pitch'] = trirotations[:,1]
data['tri_roll'] = trirotations[:,2]

# fig = px.scatter(x=data.Frame,y=data.MXS_RX,color=data.experiment,title="RX")
# fig.show()

# fig = px.scatter(x=data.Frame,y=data.MXS_RY,color=data.experiment,title="RY")
# fig.show()

# fig = px.scatter(x=data.Frame,y=data.MXS_RZ,color=data.experiment,title="RZ")
# fig.show()

# fig = px.scatter(x=data.Frame,y=data.mxs_yaw,color=data.experiment,title="Yaw")
# fig.show()

# fig = px.scatter(x=data.Frame,y=data.mxs_pitch,color=data.experiment,title="Pitch")
# fig.show()

# fig = px.scatter(x=data.Frame,y=data.mxs_roll,color=data.experiment,title="Roll")
# fig.show()

def sin_fit(x,a,b,c,d,e):
    return a * np.exp(b*(x-200)) * np.sin(2*np.pi*c*x - d) + e

def fit_sin(data,p0=None,bounds=None):
    return scipy.optimize.curve_fit(sin_fit,data.Frame,data.mxs_yaw,p0=p0,bounds=bounds)

def print_fit(fit,name):
    print(name)
    print("Amp: {}\nDecay: {}\nFreq: {}\nPhase: {}\nDC: {}".format(*fit))
    print(f"Freq {fit[2]*30} Hz\n")

def plot_fit(data,fit,name):
    fig = px.scatter(x=data.Frame/30,y=[data.mxs_yaw,sin_fit(data.Frame,*fit)],title=name)
    fig.show()


## Triangle
trianglewave = data[(data.experiment == "triangle") & (data.Frame > 350)]

# fig = px.scatter(x=trianglewave.Frame,y=[trianglewave.tri_roll,trianglewave.tri_pitch,trianglewave.tri_yaw],title="Triangle")
# fig.show()

triangle_fit,_ = scipy.optimize.curve_fit(sin_fit,trianglewave.Frame,trianglewave.tri_yaw,
    p0=[15.0,-0.001,np.pi/55,1.0,0.0],bounds=(
    [ 5.0, -np.inf,    0.0, -np.inf,  -5.0],
    [20.0,       0, np.inf,  np.inf,   5.0])
    )

# fig = px.scatter(x=trianglewave.Frame/30,y=[trianglewave.tri_yaw,sin_fit(trianglewave.Frame,*triangle_fit)],title="Triangle")
# fig.show()
print_fit(triangle_fit,"Triangle")


## Blocks
blockswave = data[(data.experiment == "blocks") & (data.Frame > 150)]

# fig = px.scatter(x=blockswave.Frame/30,y=[blockswave.tri_roll,blockswave.tri_pitch,blockswave.tri_yaw],title="Blocks")
# fig.show()

blocks_fit,_ = scipy.optimize.curve_fit(sin_fit,blockswave.Frame,blockswave.tri_yaw,
    p0=[6.0,-0.001,0.0466,1.0,0.0],bounds=(
    [ 4.0, -np.inf,    0.0, -np.inf,  -5.0],
    [10.0,       0, np.inf,  np.inf,   5.0])
    )

# fig = px.scatter(x=trianglewave.Frame/30,y=[trianglewave.tri_yaw,sin_fit(trianglewave.Frame,*triangle_fit)],title="Triangle")
# fig.show()
print_fit(blocks_fit,"Blocks")


## Yaw Battery

yawbattwave = data[(data.experiment == "yawbatt") & (data.Frame > 200)]
yawbatt_fit,_ = fit_sin(yawbattwave,p0=[2.0,-0.001,np.pi/55,1.0,94.0],bounds=(
    [1.5, -np.inf,    0.0, -np.inf,  90.0],
    [5.0,       0, np.inf,  np.inf, 100.0])
    )

# plot_fit(yawbattwave,yawbatt_fit,"YawBatt")
print_fit(yawbatt_fit,"YawBatt")


## Pitch Battery

pitchbattwave = data[(data.experiment == "pitchbatt") & (data.Frame > 200)]
pitchbatt_fit,_ = fit_sin(pitchbattwave,p0=[2.0,-0.001,np.pi/55,1.0,84.0],bounds=(
    [1.5, -np.inf,    0.0, -np.inf,  84.0],
    [5.0,       0, np.inf,  np.inf,  86.0])
    )
# plot_fit(pitchbattwave,pitchbatt_fit,"PitchBatt")
print_fit(pitchbatt_fit,"PitchBatt")


## Roll Battery

rollbattwave = data[(data.experiment == "rollbatt") & (data.Frame > 200)]
rollbatt_fit,_ = fit_sin(rollbattwave,p0=[2.0,-0.0001,0.02,1.0,-125.0],bounds=(
    [1.5, -0.01,    0.0, -np.inf, -130.0],
    [5.0,     0, np.inf,  np.inf, -120.0])
    )
# plot_fit(rollbattwave,rollbatt_fit,"RollBatt")
print_fit(rollbatt_fit,"RollBatt")


## Yaw

yawwave = data[(data.experiment == "yaw") & (data.Frame > 200)]
yaw_fit,_ = fit_sin(yawwave,p0=[2.0,-0.001,np.pi/55,1.0,94.0],bounds=(
    [1.5, -np.inf,    0.0, -np.inf,  90.0],
    [5.0,       0, np.inf,  np.inf, 100.0])
    )

# plot_fit(yawwave,yaw_fit,"Yaw")
print_fit(yaw_fit,"Yaw")


## Pitch

pitchwave = data[(data.experiment == "pitch") & (data.Frame > 200)]
pitch_fit,_ = fit_sin(pitchwave,p0=[2.0,-0.001,np.pi/55,1.0,84.0],bounds=(
    [1.5, -np.inf,    0.0, -np.inf,  84.0],
    [5.0,       0, np.inf,  np.inf,  86.0])
    )
# plot_fit(pitchwave,pitch_fit,"Pitch")
print_fit(pitch_fit,"Pitch")


## Roll

rollwave = data[(data.experiment == "roll") & (data.Frame > 200)]
roll_fit,_ = fit_sin(rollwave,p0=[2.0,-0.0001,0.02,1.0,-125.0],bounds=(
    [1.5, -0.01,    0.0, -np.inf, -130.0],
    [5.0,     0, np.inf,  np.inf, -120.0])
    )
# plot_fit(rollwave,roll_fit,"Roll")
print_fit(roll_fit,"Roll")


## Calculating inertias...

# String lengths:
#  A: 1423 mm
#  B: 1420 mm
#  C: 1405 mm
# Mean: 1416 mm -> 1.416 m
filament_length = 1.416 # m

triangle_side_length = 0.695 # m
circumcircle_radius = triangle_side_length * np.sqrt(3) / 3
inscribed_circle_radius = triangle_side_length * np.sqrt(3) / 6

hole_spacing = 0.620 # m
filament_radius = hole_spacing * np.sqrt(3) / 3

def get_freq(fit):
    return fit[2]*30

def get_inertia(freq,mass):
    L = filament_length # m
    g = 9.81 # m/s^2
    R = filament_radius # m
    # J = mgR^2/(L*(2*pi*f)^2)
    return mass * g * R**2 / (L*(2.0*np.pi*freq)**2)

## Rigging

triangle_mass = 0.344 # kg
triangle_inertia = get_inertia(get_freq(triangle_fit),triangle_mass)

print(f"Triangle inertia: {triangle_inertia}\n")

blocks_mass = 0.870 # kg
blocks_inertia = get_inertia(get_freq(blocks_fit),blocks_mass + triangle_mass)

## Plane with battery

plane_mass_with_batt = 1.418 # kg

clamp_mass = 0.178 # kg
clamp_offset = inscribed_circle_radius
clamp_inertia = clamp_mass * clamp_offset**2

pitchbatt_inertia = get_inertia(get_freq(pitchbatt_fit),plane_mass_with_batt + triangle_mass + clamp_mass) - (triangle_inertia + clamp_inertia)
rollbatt_inertia = get_inertia(get_freq(rollbatt_fit), plane_mass_with_batt + triangle_mass + blocks_mass) - (triangle_inertia + blocks_inertia)
yawbatt_inertia = get_inertia(get_freq(yawbatt_fit), plane_mass_with_batt + triangle_mass) - triangle_inertia

print(f"Plane batt inertias:\n Pitch {pitchbatt_inertia}\n Roll: {rollbatt_inertia}\n Yaw: {yawbatt_inertia}\n")

## Plane without battery

plane_mass = 1.221 # kg

pitch_inertia = get_inertia(get_freq(pitch_fit),plane_mass + triangle_mass) - triangle_inertia
roll_inertia = get_inertia(get_freq(roll_fit),plane_mass + triangle_mass + blocks_mass) - (triangle_inertia + blocks_inertia)
yaw_inertia = get_inertia(get_freq(yaw_fit),plane_mass + triangle_mass) - triangle_inertia

print(f"Plane inertias:\n Pitch {pitch_inertia}\n Roll: {roll_inertia}\n Yaw: {yaw_inertia}\n")


## Translations

triangle_pos = ( data.LowerTriangle_TX.mean(), data.LowerTriangle_TY.mean() )
print(triangle_pos)

experiments = ["yaw","pitch","roll","yawbatt","pitchbatt","rollbatt"]

for experiment in experiments:
    experiment_data = data[data.experiment == experiment]
    plane_pos = ( experiment_data.MXS_TX.mean(), experiment_data.MXS_TY.mean() )
    print(f"{experiment}: {plane_pos}")

data = data[(data.experiment == "triangle") & (data.Frame > 400)]

swing_freq = 1.0 / ( 2*np.pi*np.sqrt(filament_length/9.81) )

rot = sin_fit(data.Frame,*triangle_fit[:-1],0)

# tx_fit = sin_fit(data.Frame,  4.5, 0, swing_freq/30, 2.0, 1035) +  0.17*rot
# ty_fit = sin_fit(data.Frame, 29.0, 0, swing_freq/30, 1.7, 1045) + -0.32*rot

ox = -16
oy = -10

V_x = -1051
V_y = -1054

tx_fit = sin_fit(data.Frame,  4.5, 0, swing_freq/30, 2.0, 0) + ox*np.cos(np.radians(rot)) - oy*np.sin(np.radians(rot)) - V_x
ty_fit = sin_fit(data.Frame, 29.0, 0, swing_freq/30, 1.7, 0) + ox*np.sin(np.radians(rot)) + oy*np.cos(np.radians(rot)) - V_y

fig = px.scatter(x=data.Frame/30,y=[data.LowerTriangle_TX,data.LowerTriangle_TY, tx_fit, ty_fit])
fig.show()

import sys
sys.exit(0)

# fig = px.scatter(x=sin_fit(data.Frame,4.5,0,swing_freq/30,2.0,0),y=sin_fit(data.Frame,29,0,swing_freq/30,1.7,0))
# fig = px.scatter(x=data.LowerTriangle_TX,y=data.LowerTriangle_TY)
# fig.show()

# fig = px.scatter(
#     x=sin_fit(data.Frame,4.5,0,swing_freq/30,2.0,0) + 0.17*rot,
#     y=sin_fit(data.Frame,29,0,swing_freq/30,1.7,0) - 0.32*rot
# )
# fig.show()

# fig = px.scatter(x=0.17*rot,y=-0.32*rot)
# fig.show()

swing_major_axis = [1,0]
swing_major_amplitude = 0.5
swing_minor_amplitude = 0.2

# https://math.stackexchange.com/questions/3678977/
#
# Given: x = Asin(wt), y = Bsin(wt+p)
#
# sin(wt) = x/A
# y = B*sin(wt)*cos(p) + B*cos(wt)*sin(p)
# y = B*x/A*cos(p) + B*cos(wt)*sin(p)
#
# B*cos(wt)*sin(p) = y - B*x/A*cos(p)
# cos(wt) = (y - B*x/A*cos(p)) / (B*sin(p))
#
# Substitute results into cos^2(x) + sin^2(x) = 1
#
# (y - B*x/A*cos(p))^2 / (B*sin(p))^2 + (x/A)^2 = 1
# (y/Bsin(p) - x/Atan(p))^2 + (x/A)^2 = 1
# y^2/(Bsin(p))^2  - 2xy/(ABsin(p)tan(p)) + x^2/(Atan(p))^2 + x^2/A^2 = 1
# y^2/(Bsin(p))^2  - 2xy/(ABsin(p)tan(p)) + x^2( 1/(Atan(p))^2 + 1/A^2 ) = 1
# y^2/(Bsin(p))^2  - 2xy/(ABsin(p)tan(p)) + x^2(1+tan(p)^2)/(Atan(p))^2 = 1
#
# Just the term: (1+tan(p)^2)/(Atan(p))^2
#  = sec^2(p) / (Atan(p))^2
#  = 1/( A cos(p) tan(p) )^2
#  = 1/(A sin(p))^2
#
# Substituting back into the above...
# y^2/(Bsin(p))^2 - 2xy/(ABsin(p)tan(p)) + x^2/(Asin(p))^2 = 1
# Which is close to elliptical form
#
# https://math.stackexchange.com/questions/426150/
# For a tilted ellipse around the origin:
# (xcos(H) + ysin(H))^2/a^2 + (xsin(H) - ycos(H))^2/b^2 = 1
#
# (x^2cos(H)^2 + 2xycos(H)sin(H) + y^2sin(H)^2) / a^2 + (x^2sin(H)^2 - 2xysin(H)cos(H) + y^2cos(H)^2) / b^2 = 1
#
# x^2(cos(H)^2/a^2 + sin(H)^2/b^2) + 2xycos(H)sin(H)(1/a^2 - 1/b^2) + y^2(sin(H)^2/a^2 + cos(H)^2/b^2) = 1
#
# Equating terms:
# x^2:
#  cos(H)^2/a^2 + sin(H)^2/b^2 = 1/(Asin(p))^2
# y^2:
#  sin(H)^2/a^2 + cos(H)^2/b^2 = 1/(Bsin(p))^2
# xy:
#  cos(H)sin(H)(1/a^2 - 1/b^2) = -1/(ABsin(p)tan(p))
#
# Geogebra
# Ellipses: https://www.geogebra.org/calculator/gt2xypq5
# Params: https://www.geogebra.org/calculator/d7agzwxp
#


def residual(A,B,p,H,a,b):
    xSq = np.cos(H)**2 / a**2 + np.sin(H)**2 / b**2  -  1/(A*np.sin(p))**2
    ySq = np.sin(H)**2 / a**2 + np.cos(H)**2 / b**2  -  1/(B*np.sin(p))**2
    xy =  np.cos(H)*np.sin(H)*(1/a**2 - 1/b**2)   +  1/(A*B*np.sin(p)*np.tan(p))
    return np.sqrt(xSq**2 + ySq**2 + xy**2)

def calc_ellipse(A,B,p):
    def ellipse_residual(x):
        [H,a,b] = x
        return residual(A,B,p,H,a,b)

    #return scipy.optimize.least_squares(ellipse_residual,[np.pi/4,3.0,3.0],bounds=([-np.pi/2,0.0,0.0],[np.pi/2,np.inf,np.inf]))
    return scipy.optimize.least_squares(
        ellipse_residual,
        [1.43,30.0,1.17],
        # bounds=([-np.pi,0.0,0.0],[np.pi,np.inf,np.inf]),
        max_nfev=100000,
        method='dogbox',
        gtol=1e-15,
        xtol=1e-15,
        ftol=1e-15
        )

def calc_lissajous(H,a,b):
    def lissajous_residual(x):
        [A,B,p] = x
        return residual(A,B,p,H,a,b)

    return scipy.optimize.least_squares(lissajous_residual,[3.0,3.0,np.pi/4],bounds=([0.0,0.0,-np.pi],[np.inf,np.inf,np.pi]),max_nfev=100000)

def get_ellipse_points(H,a,b):
    t = np.linspace(0,np.pi*2,100)
    coords = zip( a * np.cos(t), b * np.sin(t) )
    sH = np.sin(H)
    cH = np.cos(H)
    
    xs = []
    ys = []
    for (x,y) in coords:
        x_p = cH*x - sH*y
        y_p = sH*x + cH*y
        xs.append(x_p)
        ys.append(y_p)
    
    return xs,ys

A = 4.5
B = 29
p = (1.7 - 2.0) % (2*np.pi)
print(p)

result = calc_ellipse(A,B,p)
print(result)

fig = px.scatter(x=sin_fit(data.Frame,4.5,0,swing_freq/30,2.0,0),y=sin_fit(data.Frame,29,0,swing_freq/30,1.7,0))
[H,a,b] = result.x
xs,ys = get_ellipse_points(H,a,b)
fig.add_scatter(x=xs,y=ys)

# result = calc_lissajous(*result.x)
# print(result)

# [A,B,p] = result.x

# t = np.linspace(1,500,num=500)
# x = sin_fit(t,A,0,swing_freq/30,0,0)
# y = sin_fit(t,B,0,swing_freq/30,np.pi/2 + p,0)

# fig.add_scatter(x=x,y=y)

# fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.show()
