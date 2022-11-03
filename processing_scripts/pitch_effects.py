import cloudpickle as pickle
import matplotlib
import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt

import utils

CREATE_LINES = True
CREATE_SURFACES = False
CREATE_MANIFOLD = False

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

from utils.fits import *

fits = {}

# Calculate C_M for neutral
fig = plt.figure()
ax = fig.add_subplot()

neutral_data = utils.select_neutral(data)
c_ms = neutral_data.load_m / utils.qSc(neutral_data.airspeed)

def c_m_curve(alpha,cm_0,alpha_cm0,cm_hscale,cm_vscale):
    alpha_lim = 15
    asymptote = 0.5
    k=12
    return S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim),k) * (cm_vscale*np.tan(cm_hscale*(alpha-alpha_cm0)) + cm_0) \
        + (1.0-H(alpha,np.radians(-alpha_lim),k)) * asymptote \
        + H(alpha,np.radians(alpha_lim),k) * -asymptote

popt,pcov = scipy.optimize.curve_fit(c_m_curve,np.radians(neutral_data.pitch),c_ms,maxfev=10000)
c_m_pitch_fit = Fit(c_m_curve,popt)

fits["c_m_pitch"] = (c_m_pitch_fit,"C_m wrt (pitch(rad))")

ax.scatter(neutral_data.pitch,c_ms,c=neutral_data.airspeed)

fit_samples = np.linspace(-30,30,200)
fitted = c_m_pitch_fit(np.radians(fit_samples))
ax.plot(fit_samples,fitted,c="red")

ax.set_xlabel("Pitch (deg)")
ax.set_ylabel("C_M")
ax.grid(True)

# Calculate C_M_elev
fig = plt.figure()
ax = fig.add_subplot()

elevdata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.rudder)<2.0)
        ]

c_m_deltas = elevdata.load_m / utils.qSc(elevdata.airspeed) - c_m_pitch_fit(np.radians(elevdata.pitch))
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(elevdata.elevator),c_m_deltas)
c_m_elev = Fit(tanh,popt)

ax.scatter(elevdata.elevator,c_m_deltas,c=elevdata.pitch)
fit_samples = np.linspace(-40,40)
fitted = c_m_elev(np.radians(fit_samples))
ax.plot(fit_samples,fitted,c="red")
ax.set_xlabel("Elevator angle (deg)")
ax.set_ylabel("C_M contribution")
ax.grid(True)

# Calculate C_M_elev with throttle

def thrust_val(throttle):
    pwm = (throttle * 1000) + 1000
    thrust = -4.120765323840711e-05 * pwm**2 + 0.14130986760422384 * pwm - 110
    #(density * airspeed * disk_area) # kg/s
    return thrust / (utils.density*(np.pi*utils.prop_rad**2))

def calculate_c_m_elev(data,optimize=True,with_wash=False):
    wash = thrust_val(data.throttle)/data.airspeed if with_wash else 0.0
    c_m_deltas = data.load_m / utils.qSc(data.airspeed + wash) - c_m_pitch_fit(np.radians(data.pitch))
    popt = None
    if optimize:
        p0 = c_m_elev.args
        popt,_ = scipy.optimize.curve_fit(tanh,np.radians(data.elevator),c_m_deltas,p0=p0,maxfev=10000)
    return c_m_deltas,popt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

pitches = [-10,-7.5,-5,-2.5,0.0,2.5,5,7.5,10]
throttles = [0.2,0.35,0.5,0.65,0.8]
airspeeds = [10.0,12.5,15.0,17.5,20.0,22.5]

cmap = plt.get_cmap('viridis')
cnorm = matplotlib.colors.Normalize(10.0,22.5)

for pitch in pitches:
    for thr in throttles:
        threlevdata_allaspd = data[
                (abs(data.throttle - thr)<0.05)
                & (abs(data.rig_pitch - pitch)<0.5)
                & (abs(data.aileron)<2.0)
                & (abs(data.rudder)<2.0)
            ]
        
        for aspd in airspeeds:
            threlevdata = threlevdata_allaspd[
                (abs(threlevdata_allaspd.airspeed - aspd)<0.05)
                ]
            
            if len(threlevdata.index) != 0:
                c_m_deltas,popt = calculate_c_m_elev(threlevdata)
                print(f"Optimised for thr/aspd/pitch:{thr:5.2f}/{aspd:5.1f}/{pitch:5.1f}: {popt}")
                thing_to_color = ax.scatter(threlevdata.elevator,threlevdata.throttle,c_m_deltas,c=threlevdata.airspeed,norm=cnorm,marker="+")
                fit_samples = np.linspace(-30,30)
                fitted = tanh(np.radians(fit_samples),*popt)
                fits[f"c_m_delta_elev@{thr}/{aspd}/{pitch}"] = (Fit(tanh,popt),"Change in C_m with wrt (elevator(rad)) at throttle/airspeed")
                ax.plot(fit_samples,[thr]*len(fit_samples),fitted,c=cmap(cnorm(aspd)))

ax.set_xlabel("Elevator angle (deg)")
ax.set_ylabel("Throttle setting")
ax.set_zlabel("C_M contribution")
fig.colorbar(thing_to_color)
#ax.legend([f"{t:4.2f}@{a}" for a in airspeeds for t in throttles])

if CREATE_SURFACES:
    # Calc C_M surfaces
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def c_m_surf(elev,thr,*args):
        if len(args) % 4 != 0:
            raise ValueError("length of args must be divisible by 4")
        poly_order_p1 = len(args) // 4
        a = P(thr,*args[0*poly_order_p1:1*poly_order_p1])
        b = P(thr,*args[1*poly_order_p1:2*poly_order_p1])
        c = P(thr,*args[2*poly_order_p1:3*poly_order_p1])
        d = P(thr,*args[3*poly_order_p1:4*poly_order_p1])
        return tanh(elev,a,b,c,d)

    def surface_least_sq(params,data,c_ms):
        c_me = c_m_surf(np.radians(data.elevator),data.throttle,*params)
        
        return np.linalg.norm(c_me - c_ms)

    import itertools

    for aspd in airspeeds:
        threlevdata = data[
                (abs(data.rig_pitch - 2.5)<0.5)
                & (abs(data.aileron)<2.0)
                & (abs(data.rudder)<2.0)
                & (abs(data.airspeed - aspd)<0.05)
            ]

        if len(threlevdata.index) != 0:
            c_m_deltas,_ = calculate_c_m_elev(threlevdata,False)
            m0 = [0.0,0.0,0.0,0.0]
            c0 = fits[f"c_m_delta_elev@0.5/{aspd}"][0].args
            x0 = list(itertools.chain(*zip(c0,m0)))
            
            bounds = [(-np.inf,np.inf)]*len(x0)
            # Set lower bound on tanh horizontal stretch
            #bounds[3] = (0.3,np.inf)
            
            res = scipy.optimize.minimize(surface_least_sq,x0,args=(threlevdata,c_m_deltas),method='Powell',options={"maxiter":100000},bounds=bounds)
            print(f"Surface optimised for aspd:{aspd:5.1f}: {res.x}")
            
            thing_to_color = ax.scatter(threlevdata.elevator,threlevdata.throttle,c_m_deltas,c=threlevdata.airspeed,norm=cnorm,marker="+")
            
            elevator_samples = np.linspace(-30,30)
            throttle_samples = np.linspace(0,0.8)
            T,E = np.meshgrid(throttle_samples,elevator_samples)
            
            fits[f"c_m_delta_elev@{aspd}"] = (Fit(c_m_surf,res.x),"Change in C_m wrt (elevator(rad),throttle) at airspeed")
            
            c_me = np.array(c_m_surf(np.radians(np.ravel(E)),np.ravel(T),*res.x))
            CM = c_me.reshape(E.shape)
            
            ax.plot_surface(E,T,CM,color=[*cmap(cnorm(aspd))[0:3],0.5])

    ax.set_xlabel("Elevator angle (deg)")
    ax.set_ylabel("Throttle setting")
    ax.set_zlabel("C_M contribution")
    #fig.colorbar(thing_to_color)

if CREATE_MANIFOLD:
    # Calc C_N manifold
    SURFACE_POLY_ORDER = 1
    MANIFOLD_POLY_ORDER = 2

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def c_m_manif(elev,thr,aspd,surface_order,*args):
        order = surface_order + 1
        if len(args) % order != 0:
            raise ValueError(f"length of args must be divisible by 4*(order+1) ({4*(order+1)})")
        poly_order_p1 = len(args) // (4*order)

        polyresults = []
        for i in range(4*order):
            polyresults.append(P(aspd,*args[i*poly_order_p1:(i+1)*poly_order_p1]))
        
        return c_m_surf(elev,thr, *polyresults)

    def manif_least_sq(params,data,c_ns):
        c_me = c_m_manif(np.radians(data.elevator),data.throttle,data.airspeed,2,*params)
        
        return np.linalg.norm(c_me - c_ns)

    threlevdata = data[
            (abs(data.rig_pitch - 2.5)<0.5)
            & (abs(data.aileron)<2.0)
            & (abs(data.rudder)<2.0)
        ]

    if len(threlevdata.index) != 0:
        c_m_deltas,_ = calculate_c_m_elev(threlevdata,False)
        #c0 = [0.0]*(SURFACE_POLY_ORDER+1)*4
        c0 = fits["c_m_delta_elev@15.0"][0].args
        # m0 = [0.0]*len(c0)
        # x0 = list(itertools.chain(*zip(c0,m0)))
        # c0 = [0.5]*4*SURFACE_POLY_ORDER
        b0 = [0.0]*len(c0)
        a0 = [0.0]*len(c0)
        x0 = list(itertools.chain(*zip(c0,b0,a0)))
        res = scipy.optimize.minimize(manif_least_sq,x0,args=(threlevdata,c_m_deltas),method="CG",options={"maxiter":100000})
        print(res)
        print(f"Manifold optimised: {res.x}")
        
        thing_to_color = ax.scatter(threlevdata.elevator,threlevdata.throttle,c_m_deltas,c=threlevdata.airspeed,norm=cnorm,marker="+")
        
        for aspd in airspeeds:
            elevator_samples = np.linspace(-30,30)
            throttle_samples = np.linspace(0,1)
            T,E = np.meshgrid(throttle_samples,elevator_samples)
            
            c_ne = np.array(c_m_manif(np.radians(np.ravel(E)),np.ravel(T),aspd,SURFACE_POLY_ORDER,*res.x))
            CM = c_ne.reshape(E.shape)
            
            ax.plot_surface(E,T,CM,color=[*cmap(cnorm(aspd))[0:3],0.5])

    ax.set_xlabel("Rudder angle (deg)")
    ax.set_ylabel("Throttle setting")
    ax.set_zlabel("C_N contribution")
    #fig.colorbar(thing_to_color)


    fits[f"c_m_delta_elev"] = (Fit(c_m_manif,[SURFACE_POLY_ORDER,*res.x]),"Change in C_m wrt (elevator(rad),throttle,airspeed)")

    manifold = fits["c_m_delta_elev"][0]

fig = plt.figure()
axs = fig.subplots(2,3)

colours = ["red","green","blue","orange","purple","brown","yellow","pink","black","grey"]
positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

for (p,pitch) in enumerate(pitches):
    for (t,thr) in enumerate(throttles):
        for (i,aspd) in enumerate(airspeeds):
            fit_key = f"c_m_delta_elev@{thr}/{aspd}/{pitch}"
            if fit_key not in fits:
                continue
            fit = fits[fit_key][0]
            
            (j,k) = positions[i]
            
            samples = np.linspace(-30,30)
            axs[j][k].plot(samples,fit(np.radians(samples)),color=colours[p])
            if CREATE_SURFACES:
                surface = fits[f"c_m_delta_elev@{aspd}"][0]
                axs[j][k].plot(samples,surface(np.radians(samples),thr),'.',color=colours[t])
            if CREATE_MANIFOLD:
                axs[j][k].plot(samples,manifold(np.radians(samples),thr,aspd),'--',color=colours[t])
            axs[j][k].set_title(f"{aspd}")

plt.show()

with open("c_m_fits.cpkl","wb") as f:
    pickle.dump(fits,f)

import yaml
with open("c_m_fits.yaml","w") as f:
    yaml.dump({ k:[float(e) for e in fit[0].args] for (k,fit) in fits.items()},f)