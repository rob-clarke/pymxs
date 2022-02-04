import math
from collections import namedtuple
from numpy.polynomial.polynomial import Polynomial

import pandas as pd
from numpy.polynomial import Polynomial

pwm_conversions = {
    'aileron_1': Polynomial([-0.00035, 0.01696, -13.5732, 1525][::-1]),
    'elevator': Polynomial([-0.00066, 0.01782, -11.4931, 1434.85][::-1]),
    'rudder': Polynomial([0.00138, -0.07209, 11.8365, 1528.14][::-1]),
    'aileron_2': Polynomial([0.01319, -0.01243, 10.7598, 1475][::-1])
}

def angle_to_pwm(angle, polynomial):
    return round(polynomial(angle))

def pwm_to_angle(pwm, polynomial):
    roots = (polynomial - pwm).roots()
    angle = (roots.real[abs(roots.imag)<1e-10])[0]
    return round(angle*100)/100

def throttle_to_pwm(throttle):
    pwm = int((throttle - 0) * (2000 - 1000) / (1 - 0) + 1000)
    return pwm

def pwm_to_throttle(pwm):
    return (pwm - 1000) / 1000.0

def calc_controls_for_row(row):
    aileron = pwm_to_angle(row.aileron1_pwm,pwm_conversions["aileron_1"])
    aileron2 = pwm_to_angle(row.aileron2_pwm,pwm_conversions["aileron_2"])
    elevator = pwm_to_angle(row.elevator_pwm,pwm_conversions["elevator"])
    throttle = pwm_to_throttle(row.throttle_pwm)
    rudder = pwm_to_angle(row.rudder_pwm,pwm_conversions["rudder"])
    return (aileron,aileron2,elevator,throttle,rudder)

def calc_controls(data):
    augmented_columns = [*data.columns,"aileron","aileron2","elevator","throttle","rudder"]
    augmented_data = pd.DataFrame(columns=augmented_columns)
    
    for row in data.itertuples(index=False):
        (a,a2,e,t,r) = calc_controls_for_row(row)
        augmented_data = augmented_data.append(pd.DataFrame([[*row,a,a2,e,t,r]],columns=augmented_columns))
    
    return augmented_data
