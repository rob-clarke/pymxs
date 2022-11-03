import pickle
import matplotlib
import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt

import utils

chord = utils.params.chord

def add_airspeed_column(df,airspeed):
    airspeed = [airspeed]*len(df.index)
    df['airspeed'] = airspeed

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

data = None
for filename,airspeed in sources:
    newdata = pickle.load(open(filename,"rb"))
    add_airspeed_column(newdata,airspeed)
    data = pd.concat([data,newdata])

def linear(x,m,c):
    return m*x + c

def quadratic(x,a,b,c):
    return a*x**2 + b*x + c

def tanh(x,a,b,c,d):
    return a * np.tanh(b*x + c) + d

def H(x,p,k=10):
    # Approx heaviside: Return ~0 for all x < p & 1 for all x > p
    # As is approx, is differentiable but there is a rolloff between regimes
    return 1/(1+np.exp(-2*k*(x-p)))

def S(x,l,h,k=10):
    # Approx heavicentre: Return ~1 for all l < x < h, else 0.0
    # As is approx, there is rolloff
    return H(x,l,k) * (1-H(x,h,k))

# Calculate C_N for neutral
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

rudddata = utils.select_neutral(data)
c_ns = rudddata.load_n / utils.qSc(rudddata.airspeed)

ax.scatter(rudddata.rudder,c_ns,c=rudddata.pitch)
ax.set_xlabel("Rudder angle (deg)")
ax.set_ylabel("C_N contribution")

# Calculate C_N_rudd
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

rudddata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.elevator)<2.0)
        ]

c_ns = rudddata.load_n / utils.qSc(rudddata.airspeed)
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(rudddata.rudder),c_ns)
C_N = {
    "rudd_a": float(popt[0]),
    "rudd_b": float(popt[1]),
    "rudd_c": float(popt[2]),
    "rudd_d": float(popt[3])
    }

ax.scatter(rudddata.rudder,c_ns,c=rudddata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_N["rudd_a"],C_N["rudd_b"],C_N["rudd_c"],C_N["rudd_d"])
ax.plot(fit_samples,fitted,c="red")
ax.set_xlabel("Rudder angle (deg)")
ax.set_ylabel("C_N contribution")
ax.grid(True)
    

def thrust_val(throttle):
    pwm = (throttle * 1000) + 1000
    thrust = -4.120765323840711e-05 * pwm**2 + 0.14130986760422384 * pwm - 110
    #(density * airspeed * disk_area) # kg/s
    return thrust / (utils.density*(np.pi*utils.prop_rad**2))

def calculate_c_n_rudd(data,optimize=True,with_wash=False):
    wash = thrust_val(data.throttle)/data.airspeed if with_wash else 0.0
    c_ns = data.load_n / utils.qSc(data.airspeed + wash)
    popt = None
    if optimize:
        p0=[v for (k,v) in C_N.items()]
        popt,_ = scipy.optimize.curve_fit(tanh,np.radians(data.rudder),c_ns,p0=p0,maxfev=10000,bounds=([-3.0,0.2,0.0,0.0],[0.0,1.5,0.8,1.5]))
    return c_ns,popt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

throttles = [0.2,0.35,0.5,0.65,0.8]
airspeeds = [10.0,12.5,15.0,17.5,20.0,22.5]

cmap = plt.get_cmap('viridis')
cnorm = matplotlib.colors.Normalize(10.0,22.5)

for thr in throttles:
    thrrudddata_allaspd = data[
            (abs(data.throttle - thr)<0.05)
            & (abs(data.rig_pitch - 2.5)<0.5)
            & (abs(data.aileron)<2.0)
            & (abs(data.elevator)<2.0)
        ]
    
    for aspd in airspeeds:
        thrrudddata = thrrudddata_allaspd[
            (abs(thrrudddata_allaspd.airspeed - aspd)<0.05)
            ]
        
        if len(thrrudddata.index) != 0:
            c_ns,popt = calculate_c_n_rudd(thrrudddata)
            print(f"Optimised for thr/aspd:{thr:5.2f}/{aspd:5.1f}: {popt}")
            thing_to_color = ax.scatter(thrrudddata.rudder,thrrudddata.throttle,c_ns,c=thrrudddata.airspeed,norm=cnorm,marker="+")
            fit_samples = np.linspace(-30,30)
            fitted = tanh(np.radians(fit_samples),*popt)
            ax.plot(fit_samples,[thr]*len(fit_samples),fitted,c=cmap(cnorm(aspd)))

ax.set_xlabel("Rudder angle (deg)")
ax.set_ylabel("Throttle setting")
ax.set_zlabel("C_N contribution")
fig.colorbar(thing_to_color)
#ax.legend([f"{t:4.2f}@{a}" for a in airspeeds for t in throttles])


# Calc C_N surfaces
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def c_n_surf(rudd,thr,am,ac,bm,bc,cm,cc,dm,dc):
    a = linear(thr,am,ac)
    b = linear(thr,bm,bc)
    c = linear(thr,cm,cc)
    d = linear(thr,dm,dc)
    return tanh(rudd,a,b,c,d)

def surface_least_sq(params,data,c_ns):
    c_ne = c_n_surf(np.radians(data.rudder),data.throttle,*params)
    
    return np.linalg.norm(c_ne - c_ns)

import itertools

for aspd in airspeeds:
    thrrudddata = data[
            (abs(data.rig_pitch - 2.5)<0.5)
            & (abs(data.aileron)<2.0)
            & (abs(data.elevator)<2.0)
            & (abs(data.airspeed - aspd)<0.05)
        ]

    if len(thrrudddata.index) != 0:
        c_ns,_ = calculate_c_n_rudd(thrrudddata,False)
        m0 = [0.0,0.0,0.0,0.0]
        c0 = [-2.9999798, 0.45542569, 0.4081167, 1.08740676]
        x0 = list(itertools.chain(*zip(m0,c0)))
        res = scipy.optimize.minimize(surface_least_sq,x0,args=(thrrudddata,c_ns),method='Powell',options={"maxiter":100000})
        print(res)
        print(f"Optimised for aspd:{aspd:5.1f}: {res.x}")
        
        thing_to_color = ax.scatter(thrrudddata.rudder,thrrudddata.throttle,c_ns,c=thrrudddata.airspeed,norm=cnorm,marker="+")
        
        rudder_samples = np.linspace(-30,30)
        throttle_samples = np.linspace(0,1)
        T,R = np.meshgrid(throttle_samples,rudder_samples)
        
        c_ne = np.array(c_n_surf(np.radians(np.ravel(R)),np.ravel(T),*res.x))
        CN = c_ne.reshape(R.shape)
        
        ax.plot_surface(R,T,CN,color=[*cmap(cnorm(aspd))[0:3],0.5])

ax.set_xlabel("Rudder angle (deg)")
ax.set_ylabel("Throttle setting")
ax.set_zlabel("C_N contribution")
#fig.colorbar(thing_to_color)

# Calc C_N manifold
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def c_n_manif(rudd,thr,aspd,amm,amc,acm,acc,bmm,bmc,bcm,bcc,cmm,cmc,ccm,ccc,dmm,dmc,dcm,dcc):
    am = linear(aspd,amm,amc)
    ac = linear(aspd,acm,acc)
    bm = linear(aspd,bmm,bmc)
    bc = linear(aspd,bcm,bcc)
    cm = linear(aspd,cmm,cmc)
    cc = linear(aspd,ccm,ccc)
    dm = linear(aspd,dmm,dmc)
    dc = linear(aspd,dcm,dcc)
    return c_n_surf(rudd,thr,am,ac,bm,bc,cm,cc,dm,dc)

def manif_least_sq(params,data,c_ns):
    c_ne = c_n_manif(np.radians(data.rudder),data.throttle,data.airspeed,*params)
    
    return np.linalg.norm(c_ne - c_ns)

thrrudddata = data[
        (abs(data.rig_pitch - 2.5)<0.5)
        & (abs(data.aileron)<2.0)
        & (abs(data.elevator)<2.0)
    ]

if len(thrrudddata.index) != 0:
    c_ns,_ = calculate_c_n_rudd(thrrudddata,False)
    c0 = [-0.13642334, -2.8655547, 0.28847784, 0.26922272, 0.00860537, 0.40901526, -0.01916245, 1.10104976]
    m0 = [0.0]*len(c0)
    x0 = list(itertools.chain(*zip(m0,c0)))
    res = scipy.optimize.minimize(manif_least_sq,x0,args=(thrrudddata,c_ns),method='Powell',options={"maxiter":100000})
    print(res)
    print(f"Optimised for aspd:{aspd:5.1f}: {res.x}")
    
    thing_to_color = ax.scatter(thrrudddata.rudder,thrrudddata.throttle,c_ns,c=thrrudddata.airspeed,norm=cnorm,marker="+")
    
    for aspd in airspeeds:
        rudder_samples = np.linspace(-30,30)
        throttle_samples = np.linspace(0,1)
        T,R = np.meshgrid(throttle_samples,rudder_samples)
        
        c_ne = np.array(c_n_manif(np.radians(np.ravel(R)),np.ravel(T),aspd,*res.x))
        CN = c_ne.reshape(R.shape)
        
        ax.plot_surface(R,T,CN,color=[*cmap(cnorm(aspd))[0:3],0.5])

ax.set_xlabel("Rudder angle (deg)")
ax.set_ylabel("Throttle setting")
ax.set_zlabel("C_N contribution")
#fig.colorbar(thing_to_color)

plt.show()
