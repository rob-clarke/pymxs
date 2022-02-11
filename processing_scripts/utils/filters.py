def select_neutral(data,throttle=0.0):
    return data[
        (abs(data.aileron)<2.0)
        & (abs(data.elevator)<2.0)
        & (abs(data.throttle-throttle)<0.05)
        & (abs(data.rudder)<2.0)
        ]

def select_elev(data,elev=None,epsilon=2.0):
    if elev is None:
        return data[(abs(data.aileron)<2.0) & (abs(data.rudder)<2.0)]
    else:
        return data[
            (abs(data.aileron)<2.0)
            & (abs(data.rudder)<2.0)
            & (abs(data.elevator - elev)<epsilon)
            ]
