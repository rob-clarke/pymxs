import pickle
import matplotlib
import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt

import utils

chord = 0.23

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


# Calculate C_lift
def c_l_curve(alpha,cl_0,cl_alpha,pstall,nstall):
    #astall = np.radians(10)
    return S(alpha,nstall,pstall) * (cl_0 + cl_alpha * alpha) # Linear regime
    
neutral_data = utils.select_neutral(data)

c_lifts = neutral_data.lift / utils.qS(neutral_data.airspeed)

popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_lifts)
C_lift = {
    "zero": float(popt[1]),
    "alpha": float(popt[0])
    }
popt,pcov = scipy.optimize.curve_fit(c_l_curve,np.radians(neutral_data.pitch),c_lifts,[C_lift["zero"],C_lift["alpha"],np.radians(10),-np.radians(10)])
C_lift_complex = popt

# Calculate C_drag
def c_d_curve(alpha,cd_0,cd_alpha,alpha_cd0):
    alpha_lim = 30
    return S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim)) * (cd_alpha*(alpha-alpha_cd0)**2 + cd_0) \
        + (1.0-S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim))) * 2.0
    
c_ds = neutral_data.drag / utils.qS(neutral_data.airspeed)

popt,pcov = scipy.optimize.curve_fit(quadratic,np.radians(neutral_data.pitch),c_ds)
C_drag = {
    "zero": float(popt[2]),
    "alpha": float(popt[1]),
    "alpha2": float(popt[0])
    }

popt,pcov = scipy.optimize.curve_fit(c_d_curve,np.radians(neutral_data.pitch),c_ds,maxfev=10000)
C_drag_complex = popt

# Calculate C_M
def c_m_curve(alpha,cm_0,alpha_cm0,cm_hscale,cm_vscale):
    alpha_lim = 15
    asymptote = 0.5
    k=12
    return S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim),k) * (cm_vscale*np.tan(cm_hscale*(alpha-alpha_cm0)) + cm_0) \
        + (1.0-H(alpha,np.radians(-alpha_lim),k)) * asymptote \
        + H(alpha,np.radians(alpha_lim),k) * -asymptote
    
c_ms = neutral_data.load_m / utils.qSc(neutral_data.airspeed)

popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_ms)
C_M = {
    "zero": float(popt[1]),
    "alpha": float(popt[0])
    }

popt,pcov = scipy.optimize.curve_fit(c_m_curve,np.radians(neutral_data.pitch),c_ms,maxfev=10000)
C_M_complex = popt

# Make plots
fig,axs = plt.subplots(1,3)

# Plot lift
axs[0].scatter(neutral_data.pitch,c_lifts)

fit_samples = np.linspace(-15,15)
fitted = linear(np.radians(fit_samples),C_lift["alpha"],C_lift["zero"])
axs[0].plot(fit_samples,fitted,c="red")

fit_samples = np.linspace(-30,30,200)
fitted = c_l_curve(np.radians(fit_samples),*C_lift_complex)
axs[0].plot(fit_samples,fitted,c="green")

axs[0].set_xlabel("Pitch angle (deg)")
axs[0].set_ylabel("C_L")

# Plot drag
axs[1].scatter(neutral_data.pitch,c_ds)

fit_samples = np.linspace(-15,15)
fitted = quadratic(np.radians(fit_samples),C_drag["alpha2"],C_drag["alpha"],C_drag["zero"])
axs[1].plot(fit_samples,fitted,c="red")

fit_samples = np.linspace(-30,30,200)
fitted = c_d_curve(np.radians(fit_samples),*C_drag_complex)
axs[1].plot(fit_samples,fitted,c="green")

axs[1].set_xlabel("Pitch angle (deg)")
axs[1].set_ylabel("C_D")

# Plot C_M
axs[2].scatter(neutral_data.pitch,c_ms)

fit_samples = np.linspace(-15,15)
fitted = linear(np.radians(fit_samples),C_M["alpha"],C_M["zero"])
axs[2].plot(fit_samples,fitted,c="red")

fit_samples = np.linspace(-30,30,200)
fitted = c_m_curve(np.radians(fit_samples),*C_M_complex)
axs[2].plot(fit_samples,fitted,c="green")

axs[2].set_xlabel("Pitch angle (deg)")
axs[2].set_ylabel("C_M")

for ax in axs:
    ax.grid(True,'both')

# C_M_elev
elevdata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.rudder)<2.0)
        ]
c_m_alpha = linear(np.radians(elevdata.pitch),C_M["alpha"],C_M["zero"])
c_ms = elevdata.load_m / utils.qSc(elevdata.airspeed) - c_m_alpha
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(elevdata.elevator),c_ms)
C_M["elev_a"] = float(popt[0])
C_M["elev_b"] = float(popt[1])
C_M["elev_c"] = float(popt[2])
C_M["elev_d"] = float(popt[3])


fig,axs = plt.subplots(2,3)

axs[0,0].scatter(elevdata.elevator,c_ms,c=elevdata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_M["elev_a"],C_M["elev_b"],C_M["elev_c"],C_M["elev_d"])
axs[0,0].scatter(fit_samples,fitted,c="red")
axs[0,0].set_xlabel("Elevator angle (deg)")
axs[0,0].set_ylabel("C_M contribution")

# Calculate C_L_ail
aildata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.elevator)<2.0)
        & (abs(data.rudder)<2.0)
        ]

c_ls = aildata.load_l / utils.qSc(aildata.airspeed)
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(aildata.aileron),c_ls)
C_L = {
    "ail_a": float(popt[0]),
    "ail_b": float(popt[1]),
    "ail_c": float(popt[2]),
    "ail_d": float(popt[3])
    }

axs[0,1].scatter(aildata.aileron,c_ls,c=aildata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_L["ail_a"],C_L["ail_b"],C_L["ail_c"],C_L["ail_d"])
axs[0,1].scatter(fit_samples,fitted,c="red")
axs[0,1].set_xlabel("Aileron angle (deg)")
axs[0,1].set_ylabel("C_L contribution")

# Calculate C_L_rudd
rudddata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.elevator)<2.0)
        ]

c_ls = rudddata.load_l / utils.qSc(rudddata.airspeed)
popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(rudddata.rudder),c_ls)
C_L["rudd_a"] = float(popt[0])
C_L["rudd_b"] = float(popt[1])
C_L["rudd_c"] = float(popt[2])
C_L["rudd_d"] = float(popt[3])

axs[1,1].scatter(rudddata.rudder,c_ls,c=rudddata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_L["rudd_a"],C_L["rudd_b"],C_L["rudd_c"],C_L["rudd_d"])
axs[1,1].scatter(fit_samples,fitted,c="red")
axs[1,1].set_xlabel("Rudder angle (deg)")
axs[1,1].set_ylabel("C_L contribution")

# Calculate C_N_rudd
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

axs[0,2].scatter(rudddata.rudder,c_ns,c=rudddata.pitch)
fit_samples = np.linspace(-30,30)
fitted = tanh(np.radians(fit_samples),C_N["rudd_a"],C_N["rudd_b"],C_N["rudd_c"],C_N["rudd_d"])
axs[0,2].plot(fit_samples,fitted,c="red")
axs[0,2].set_xlabel("Rudder angle (deg)")
axs[0,2].set_ylabel("C_N contribution")

for ax in axs.flatten():
    ax.grid(True,'both')
    

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

for ax in axs.flatten():
    ax.grid(True,'both')


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


# Calculate C_T
thrdata = data[
        (abs(data.aileron)<2.0)
        & (data.throttle > 0.1)
        & (abs(data.elevator)<2.0)
        & (abs(data.rudder)<2.0)
        ]

# c_lift = linear(np.radians(thrdata.pitch),C_lift["alpha"],C_lift["zero"])
# c_drag = quadratic(np.radians(thrdata.pitch),C_drag["alpha2"],C_drag["alpha"],C_drag["zero"])
c_lift = c_l_curve(np.radians(thrdata.pitch),*C_lift_complex)
c_drag = c_d_curve(np.radians(thrdata.pitch),*C_drag_complex)

lift = c_lift * utils.qS(thrdata.airspeed)
drag = c_drag * utils.qS(thrdata.airspeed)

# [F_x] = [[ cos(a) -sin(a) ]] [-D]
# [F_z]   [[ sin(a)  cos(a) ]] [-L]
expected_load_x = -drag * np.cos(np.radians(thrdata.pitch)) + lift * np.sin(np.radians(thrdata.pitch))
thrust = thrdata.load_x - expected_load_x

fig, axs = plt.subplots(1,2)
d = axs[0].scatter(thrdata.pitch,thrust,c=thrdata.airspeed)
axs[0].set_xlabel("Pitch angle (deg)")
axs[0].set_ylabel("Thrust (N)")
cbar = fig.colorbar(d,ax=axs[0])
cbar.set_label("Airspeed (m/s)")

rho = 1.225
D = 0.28
n = thrdata.rpm / 60.0

k_t = thrust / (rho * n**2 * D**4)

k_t_select = (thrust > 1.0) & (thrdata.rpm > 100)

d = axs[1].scatter(thrdata[k_t_select].pitch,k_t[k_t_select],c=thrdata[k_t_select].throttle)
axs[1].set_xlabel("Pitch angle (deg)")
axs[1].set_ylabel("Thrust coefficient")
cbar = fig.colorbar(d,ax=axs[1])
cbar.set_label("Throttle setting")

import yaml
print(yaml.dump({
    "C_lift": C_lift,
    "C_drag": C_drag,
    "C_L": C_L,
    "C_M": C_M,
    "C_N": C_N,
    }))

plt.show()