import os

import pandas as pd
import matplotlib.pyplot as plt

def get_rigpitch(filename):
    rigpitch_str = (filename.split('_')[-1])[:-4]
    negative = False
    if 'm' in rigpitch_str:
        rigpitch_str = rigpitch_str[1:]
        negative = True
    value = float(rigpitch_str)
    return -value if negative else value

def augment_with_rigpitch(data,rigpitch):
    augmented_columns = ["index",*data.columns,"rig_pitch"]
    augmented_data = pd.DataFrame(columns=augmented_columns)
    
    for row in data.itertuples():
        augmented_data = augmented_data.append(pd.DataFrame([[*row,rigpitch]],columns=augmented_columns))
    
    return augmented_data

def augment_with_airspeed(df,airspeed):
    airspeed = [airspeed]*len(df.index)
    df['airspeed'] = airspeed

def load_dir(dirpath,add_rigpitch=True):
    def load_file(dirpath,filename):
        data = pd.read_csv(os.path.join(dirpath,filename))
        if add_rigpitch:
            rigpitch = get_rigpitch(filename)
            data = augment_with_rigpitch(data,rigpitch)
        return data
            
    files = os.listdir(dirpath)
    dfs = map(lambda f: load_file(dirpath,f),files)
    return pd.concat(dfs)

from . import tares, controls

from .clean import transform_data, calc_lift_drag, shift_data
from .controls import calc_controls

def process_dir(dirpath,tare):
    raw_data = load_dir(dirpath)
    tared_data = tare.apply_tare_funcs(raw_data)
    aligned_data = shift_data(transform_data(tared_data))
    augmented_data = calc_lift_drag(calc_controls(aligned_data))
    return augmented_data

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

from .filters import *
from .params import *

density = 1.225
temperature = 293.5

def qS(airspeed,density=density):
    S = 2.625E+05 / 1000**2
    return 0.5 * density * airspeed**2 * S

def qSc(airspeed,density=density):
    return qS(airspeed,density) * 0.23

def air_viscosity(temperature):
    return 2.791e-7 * pow(temperature,0.7355)

def reynolds(airspeed,temperature=temperature,density=density):
    return density * airspeed * chord / air_viscosity(temperature)