import math
import numpy as np
import pandas as pd

def transform_data(data):
    # As mounted in aircraft, load cell convention is:
    # x+ -> Towards left wing
    # y+ -> Towards rear of aircraft
    # z+ -> Towards top of aircraft
    # l+ -> Nose down
    # m+ -> Left wing down
    # n+ -> Nose left
    
    # Standard convention is:
    # x+ -> Towards nose
    # y+ -> Towards right wing
    # z+ -> Towards bottom of aircraft
    # l+ -> Right wing down
    # m+ -> Nose up
    # n+ -> Nose right

    data_out = data.copy()
    data_out.load_x = -data.load_y
    data_out.load_y = -data.load_x
    data_out.load_z = -data.load_z
    
    data_out.load_l = -data.load_m
    data_out.load_m = -data.load_l
    data_out.load_n = -data.load_n
    
    data_out.pitch = -data.pitch
    
    return data_out

def shift_data(data,offset=[0.015,0.0,-0.023]):
    # offset: vector from load cell reference to vehicle reference point
    offset_x = offset[0]
    offset_y = offset[1]
    offset_z = offset[2]
    
    data_out = data.copy()

    # M' = M + F.cross(offset)
    data_out.load_l = data.load_l + data.load_y * offset_z - data.load_z * offset_y
    data_out.load_m = data.load_m + data.load_z * offset_x - data.load_x * offset_z
    data_out.load_n = data.load_n + data.load_x * offset_y - data.load_y * offset_x
    
    return data_out

def calc_lift_drag(data):
    augmented_columns = ["index",*data.columns,"lift","drag"]
    augmented_data = pd.DataFrame(columns=augmented_columns)
    
    # SUA: T&P pp 49
    # [F_x] = [[ cos(a) -sin(a) ]] [-D]
    # [F_z]   [[ sin(a)  cos(a) ]] [-L]
    #
    # \therefore{}
    #
    # [-D] = [[  cos(a)  sin(a) ]] [F_x]
    # [-L]   [[ -sin(a)  cos(a) ]] [F_z]
    
    for row in data.itertuples():
        pitch = np.radians(row.rig_pitch)
        drag = -(row.load_x * math.cos(pitch) + row.load_z * math.sin(pitch))
        lift = -(-row.load_x * math.sin(pitch) + row.load_z * math.cos(pitch))
        augmented_data = augmented_data.append(pd.DataFrame([[*row,lift,drag]],columns=augmented_columns))
    
    return augmented_data