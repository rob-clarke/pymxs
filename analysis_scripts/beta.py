# Do some nonsense to make imports work...
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

from common import load_beta_data, load_prop_beta_data, load_noprop_beta_data, make_plot

from processing_scripts import utils

from processing_scripts.utils.fits import linear, quadratic, H, S, Fit

import os

import numpy as np
import scipy

import matplotlib.pyplot as plt

def plot_ab(data,datasel,label,beta_only=False):
    fig = plt.figure()
    if beta_only:
        ax = fig.add_subplot()
        if type(datasel) is str:
            d = ax.scatter(data.rig_yaw, data[datasel], c=data.airspeed)
        else:
            d = ax.scatter(data.rig_yaw, datasel(data), c=data.airspeed)
        plt.grid()
        plt.xlabel("Beta (deg)")
        plt.ylabel(label)
    else:
        ax = fig.add_subplot(projection='3d')
        if type(datasel) is str:
            d = ax.scatter(data.rig_pitch, data.rig_yaw, data[datasel], c=data.airspeed)
        else:
            d = ax.scatter(data.rig_pitch, data.rig_yaw, datasel(data), c=data.airspeed)
        plt.xlabel("Alpha (deg)")
        plt.ylabel("Beta (deg)")
        ax.set_zlabel(label)
    plt.colorbar(d,ax=ax)
    return ax

if __name__ == "__main__":
    data = load_noprop_beta_data()
    # data = load_beta_data()
    # data = load_prop_beta_data()
    
    #data = data[data.rig_pitch >= 0]
    data = data[
        (abs(data.aileron)<2.0)
        & (abs(data.elevator)<2.0)
        & (abs(data.rudder)<2.0)
        & (data.throttle<0.1)
        & (data.rpm < 200)
        ]
    
    # plot_ab(data,"load_x","X load (N)")
    # plot_ab(data,"load_y","Y load (N)")
    # plot_ab(data,"load_z","Z load (N)")
    
    # plot_ab(data,"load_l","L load (Nm)")
    # plot_ab(data,"load_m","M load (Nm)")
    # plot_ab(data,"load_n","N load (Nm)")

    plot_ab(data,"lift","Lift (N)",beta_only=True)
    plot_ab(data,"drag","Drag (N)",beta_only=True)
    plot_ab(data,"sideforce","Sideforce (N)",beta_only=True)
    
    from processing_scripts.utils.clean import _get_windaxis_forces
    
    offset_forces = np.zeros((len(data.index),3))
    for (i,row) in enumerate(data.itertuples()):
        offset_forces[i,:] = _get_windaxis_forces(row,pitch_offset=-3)
    
    plot_ab(data,lambda _: -offset_forces[:,2],"Lift* (N)",beta_only=True)
    plot_ab(data,lambda _: -offset_forces[:,0],"Drag* (N)",beta_only=True)
    plot_ab(data,lambda _: offset_forces[:,1],"Sideforce* (N)",beta_only=True)
    
    def qS(aspd):
        return 0.5*1.225*np.power(aspd,2)*utils.params.S
    
    plot_ab(data,lambda d: data.lift / qS(d.airspeed),"C_L",beta_only=True)
    plot_ab(data,lambda d: data.drag / qS(d.airspeed),"C_D",beta_only=True)
    plot_ab(data,lambda d: data.sideforce / qS(d.airspeed),"C_Y",beta_only=True)
    
    # plot_ab(data,"rpm","RPM")
    
    print(max(list(data.throttle)))
    
    plt.show()

# Have X,Y,Z, L,M,N as outputs
# Have ail,ele,thr,rud, alpha,beta,aspd as inputs
# Ideally have F = f(alpha,beta,aspd) + 0.5*rho*V^2 * g(ail,ele,thr,rud)