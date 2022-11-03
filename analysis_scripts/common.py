import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pickle
import pandas as pd

import processing_scripts.utils as utils

def _load_data(sources):
    thisfiledir = os.path.dirname(os.path.abspath(__file__))
    
    data = None
    
    for filename,airspeed in sources:
        with open(thisfiledir+"/../wind_tunnel_data/processed_corrected/"+filename,"rb") as f:
            newdata = pickle.load(f)
        newdata.index = newdata["index"].iloc[:,-1]
        newdata = newdata.drop("index",axis="columns")
        utils.augment_with_airspeed(newdata,airspeed)
        data = pd.concat([data,newdata])
    
    return data


def load_data():
    sources = [
        ("data_17_10.pkl",10),
        ("data_17_12.5.pkl",12.5),
        ("data_17_15.pkl",15),
        ("data_17_17.5.pkl",17.5),
        ("data_17_20.pkl",20),
        ("data_17_22.5.pkl",22.5),
        ("data_18_10.pkl",10),
        ("data_18_15.pkl",15),
        ("data_18_20.pkl",20)
        ]
    return _load_data(sources)


def load_beta_data():
    sources = [
        ("data_05_10.pkl",10),
        ("data_05_12.5.pkl",12.5),
        ("data_05_15.pkl",15),
        ("data_05_17.5.pkl",17.5),
        ("data_05_20.pkl",20),
        
        ("data_05_10p.pkl",10),
        ("data_06_16.5.pkl",16.5),
        ("data_07_13.5.pkl",13.5),
        ]
    return _load_data(sources)

def load_prop_beta_data():
    sources = [
        ("data_05_10p.pkl",10),
        ("data_06_16.5.pkl",16.5),
        ("data_07_13.5.pkl",13.5),
        ]
    return _load_data(sources)

def load_noprop_beta_data():
    sources = [
        ("data_05_10.pkl",10),
        ("data_05_12.5.pkl",12.5),
        ("data_05_15.pkl",15),
        ("data_05_17.5.pkl",17.5),
        ("data_05_20.pkl",20)
        ]
    return _load_data(sources)


def make_plot(ax,xs,ys,samples1,fit1,samples2,fit2,xlabel,ylabel,grid=True):
    ax.scatter(xs,ys,s=1,c='black')
    ax.plot(samples1,fit1(samples1),c="red")
    if fit2 is not None:
        ax.plot(samples2,fit2(samples2),c="green")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid,'both')
