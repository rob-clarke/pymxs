# Overview

This folder contains the raw wind tunnel data for the MXS.

Files are typically named in the format: `{name}_{airspeed}_{rig_pitch}`.

Airspeed control was generally relatively relaxed (tunnel was faulty) so error bars on airspeed may need to be wide.

## PWM mapping

For all runs, PWM to surface angle mapping is as below:

```py
from numpy.polynomial import Polynomial

pwm_conversions = {
    'aileron_1': Polynomial([-0.00035, 0.01696, -13.5732, 1525][::-1]),
    'elevator': Polynomial([-0.00066, 0.01782, -11.4931, 1434.85][::-1]),
    'rudder': Polynomial([0.00138, -0.07209, 11.8365, 1528.14][::-1]),
    'aileron_2': Polynomial([0.01319, -0.01243, 10.7598, 1475][::-1])
}

def pwm_to_angle(pwm, polynomial):
    roots = (polynomial - pwm).roots()
    angle = (roots.real[abs(roots.imag)<1e-10])[0]
    return round(angle*100)/100
```

Aileron1 (Right)

Aileron2 (Left)

## 2021-11-03 & 2021-11-04

These folders contain data from static tests conducted on a thrust stand in the tunnel return section.

## 2021-11-16

See notes.txt for order of runs (under day 1)

`as10`, `as12.5` and `as15` contain data from 10m/s, 12.5m/s and 15m/s respectively.

## 2021-11-17

See notes.txt for order of runs (under day 2)

## 2021-11-18

Data with prop removed. Tare at start and likely at end of day
