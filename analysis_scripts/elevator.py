# Do some nonsense to make imports work...
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

from common import load_data, make_plot

from processing_scripts import utils

from processing_scripts.utils.fits import poly_manifold, poly_surface, tanh, Fit

import cloudpickle as pickle
import numpy as np
import scipy
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

def filter_elev_data(data):
    return data[
            (abs(data.throttle)<0.05)
            & (abs(data.aileron)<2.0)
            & (abs(data.rudder)<2.0)
            ]

def calculate_C_M_delta_elev(data,c_m_pitch_fit):
    # Calculate and return C_M_delta_elev fit
    elevdata = filter_elev_data(data)

    c_m_deltas = elevdata.load_m / utils.qSc(elevdata.airspeed) - c_m_pitch_fit(np.radians(elevdata.pitch))
    popt,_ = scipy.optimize.curve_fit(tanh,np.radians(elevdata.elevator),c_m_deltas)
    return Fit(tanh,popt)

def plot_C_M_delta_elev(elevdata,c_m_pitch_fit,c_m_delta_elev_fit):
    fig = plt.figure()
    ax = fig.add_subplot()
    
    c_m_deltas = elevdata.load_m / utils.qSc(elevdata.airspeed) - c_m_pitch_fit(np.radians(elevdata.pitch))
    ax.scatter(elevdata.elevator,c_m_deltas,c=elevdata.pitch)
    fit_samples = np.linspace(-40,40)
    fitted = c_m_delta_elev_fit(np.radians(fit_samples))
    ax.plot(fit_samples,fitted,c="red")
    ax.set_xlabel("Elevator angle (deg)")
    ax.set_ylabel("C_M contribution")
    ax.grid(True)


pitches = [-10,-7.5,-5,-2.5,0.0,2.5,5,7.5,10]
throttles = [0.0,0.2,0.35,0.5,0.65,0.8]
airspeeds = [10.0,12.5,15.0,17.5,20.0,22.5]

def calculate_c_m_elev(data,c_m_pitch_fit,c_m_delta_elev_fit,optimize=True,with_wash=False):
    print(f"{len(data.index)=}")
    def thrust_val(throttle):
        pwm = (throttle * 1000) + 1000
        thrust = -4.120765323840711e-05 * pwm**2 + 0.14130986760422384 * pwm - 110
        #(density * airspeed * disk_area) # kg/s
        return thrust / (utils.density*(np.pi*utils.prop_rad**2))
    
    wash = thrust_val(data.throttle)/data.airspeed if with_wash else 0.0
    c_m_deltas = data.load_m / utils.qSc(data.airspeed + wash) - c_m_pitch_fit(np.radians(data.pitch))
    popt = None
    if optimize:
        p0 = c_m_delta_elev_fit.args
        popt,_ = scipy.optimize.curve_fit(tanh,np.radians(data.elevator),c_m_deltas,p0=p0,maxfev=10000)
    return c_m_deltas,popt

def calculate_C_M_delta_elev_thr(data,c_m_pitch_fit,c_m_delta_elev_fit):
    # Calculate C_M_elev with throttle    
    fits = {}
    
    for pitch in pitches:
        for thr in throttles:
            # Select data for this operating point
            threlevdata_allaspd = data[
                    (abs(data.throttle - thr)<0.05)
                    & (abs(data.rig_pitch - pitch)<0.5)
                    & (abs(data.aileron)<2.0)
                    & (abs(data.rudder)<2.0)
                ]
            
            for aspd in airspeeds:
                # Select data for a particular airspeed
                threlevdata = threlevdata_allaspd[
                    (abs(threlevdata_allaspd.airspeed - aspd)<0.05)
                    ]
                
                if len(threlevdata.index) == 0:
                    # No data for these conditions
                    continue
                
                c_m_deltas,popt = calculate_c_m_elev(threlevdata,c_m_pitch_fit,c_m_delta_elev_fit)
                #print(f"Optimised for thr/aspd/pitch:{thr:5.2f}/{aspd:5.1f}/{pitch:5.1f}: {popt}")
                fits[f"c_m_delta_elev@{thr}/{aspd}/{pitch}"] = (Fit(tanh,popt),"Change in C_m with wrt (elevator(rad)) at throttle/airspeed/pitch")
    
    return fits

def plot_C_M_delta_elev_thr(data,c_m_pitch_fit,c_m_delta_elev_thr_fits):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
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
                
                if len(threlevdata.index) == 0:
                    # No data for these conditions
                    continue
                
                c_m_deltas,_ = calculate_c_m_elev(threlevdata,c_m_pitch_fit,None,False)
                thing_to_color = ax.scatter(threlevdata.elevator,threlevdata.throttle,c_m_deltas,c=threlevdata.airspeed,norm=cnorm,marker="+")
                fit_samples = np.linspace(-35,35)
                fitted = c_m_delta_elev_thr_fits[f"c_m_delta_elev@{thr}/{aspd}/{pitch}"][0](np.radians(fit_samples))
                ax.plot(fit_samples,[thr]*len(fit_samples),fitted,c=cmap(cnorm(aspd)))

    ax.set_xlabel("Elevator angle (deg)")
    ax.set_ylabel("Throttle setting")
    ax.set_zlabel("C_M contribution")
    cbar = fig.colorbar(thing_to_color)
    cbar.set_label("Airspeed (m/s)")
    #ax.legend([f"{t:4.2f}@{a}" for a in airspeeds for t in throttles])

def c_m_surf(elev,thr,*args):
    return poly_surface(elev,thr,3,*args)
    if len(args) % 4 != 0:
        raise ValueError("length of args must be divisible by 4")
    poly_order_p1 = len(args) // 4
    a = P(thr,*args[0*poly_order_p1:1*poly_order_p1])
    b = P(thr,*args[1*poly_order_p1:2*poly_order_p1])
    c = P(thr,*args[2*poly_order_p1:3*poly_order_p1])
    d = P(thr,*args[3*poly_order_p1:4*poly_order_p1])
    return tanh(elev,a,b,c,d)

def calculate_C_M_delta_elev_thr_surfaces(data,c_m_pitch_fit):
    # Calc C_M surfaces    

    def surface_least_sq(params,data,c_ms):
        c_me = c_m_surf(np.radians(data.elevator),data.throttle,*params)
        return np.linalg.norm(c_me - c_ms)

    import itertools

    fits = {}

    for aspd in airspeeds:
        # Select data for this airspeed
        threlevdata = data[
                (abs(data.rig_pitch - 2.5)<0.5)
                & (abs(data.aileron)<2.0)
                & (abs(data.rudder)<2.0)
                & (abs(data.airspeed - aspd)<0.05)
            ]

        if len(threlevdata.index) == 0:
            continue
        
        c_m_deltas,_ = calculate_c_m_elev(threlevdata,c_m_pitch_fit,None,False)
        
        # m0 = [0.0,0.0,0.0,0.0]
        # c0 = fits[f"c_m_delta_elev@0.5/{aspd}"][0].args
        # x0 = list(itertools.chain(*zip(c0,m0)))
        
        x0 = [1.0]*10
        
        bounds = [(-np.inf,np.inf)]*len(x0)
        # Set lower bound on tanh horizontal stretch
        #bounds[3] = (0.3,np.inf)
        
        res = scipy.optimize.minimize(surface_least_sq,x0,args=(threlevdata,c_m_deltas),method='Powell',options={"maxiter":100000},bounds=bounds)
        #print(f"Surface optimised for aspd:{aspd:5.1f}: {res.x}")
        
        fits[f"c_m_delta_elev@{aspd}"] = (Fit(c_m_surf,res.x),"Change in C_m wrt (elevator(rad),throttle) at airspeed")
    
    return fits

def plot_C_M_delta_elev_thr_surfaces(data,c_m_pitch_fit,c_m_delta_elev_thr_surface_fits):
    # Plot C_M surfaces
    cmap = plt.get_cmap('viridis')
    cnorm = matplotlib.colors.Normalize(10.0,22.5)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for aspd in airspeeds:
        threlevdata = data[
                (abs(data.rig_pitch - 2.5)<0.5)
                & (abs(data.aileron)<2.0)
                & (abs(data.rudder)<2.0)
                & (abs(data.airspeed - aspd)<0.05)
            ]

        if len(threlevdata.index) == 0:
            continue
        
        c_m_deltas,_ = calculate_c_m_elev(threlevdata,c_m_pitch_fit,None,False)
                
        thing_to_color = ax.scatter(threlevdata.elevator,threlevdata.throttle,c_m_deltas,c=threlevdata.airspeed,norm=cnorm,marker="+")
        
        elevator_samples = np.linspace(-35,35)
        throttle_samples = np.linspace(0,0.8)
        T,E = np.meshgrid(throttle_samples,elevator_samples)
        
        c_me = np.array(c_m_delta_elev_thr_surface_fits[f"c_m_delta_elev@{aspd}"][0](np.radians(np.ravel(E)),np.ravel(T)) )
        CM = c_me.reshape(E.shape)
        
        ax.plot_surface(E,T,CM,color=[*cmap(cnorm(aspd))[0:3],0.5])

    ax.set_xlabel("Elevator angle (deg)")
    ax.set_ylabel("Throttle setting")
    ax.set_zlabel("C_M contribution")
    cbar = fig.colorbar(thing_to_color)
    cbar.set_label("Airspeed")

def c_m_manif(elev,thr,aspd,surface_order,*args):
    return poly_manifold(elev,thr,aspd,2,*args)
    order = surface_order + 1
    if len(args) % order != 0:
        raise ValueError(f"length of args must be divisible by 4*(order+1) ({4*(order+1)})")
    poly_order_p1 = len(args) // (4*order)

    polyresults = []
    for i in range(4*order):
        polyresults.append(P(aspd,*args[i*poly_order_p1:(i+1)*poly_order_p1]))
    
    return c_m_surf(elev,thr, *polyresults)

def calculate_C_M_delta_elev_thr_manifold(data, c_m_pitch_fit):
    # Calc C_M manifold
    
    def manif_least_sq(params,data,c_ms):
        c_me = c_m_manif(np.radians(data.elevator),data.throttle,data.airspeed,None,*params)
        
        return np.linalg.norm(c_me - c_ms)

    threlevdata = data[
            (abs(data.rig_pitch - 2.5)<0.5)
            & (abs(data.aileron)<2.0)
            & (abs(data.rudder)<2.0)
        ]

    if len(threlevdata.index) == 0:
        return None
    
    c_m_deltas,_ = calculate_c_m_elev(threlevdata,c_m_pitch_fit,None,False)
    # #c0 = [0.0]*(SURFACE_POLY_ORDER+1)*4
    # c0 = fits["c_m_delta_elev@15.0"][0].args
    # # m0 = [0.0]*len(c0)
    # # x0 = list(itertools.chain(*zip(c0,m0)))
    # # c0 = [0.5]*4*SURFACE_POLY_ORDER
    # b0 = [0.0]*len(c0)
    # a0 = [0.0]*len(c0)
    # x0 = list(itertools.chain(*zip(c0,b0,a0)))
    x0 = [0.01]*21
    res = scipy.optimize.minimize(manif_least_sq,x0,args=(threlevdata,c_m_deltas),method="Powell",options={"maxiter":100000})

    return Fit(c_m_manif,[None,*res.x]),res


def plot_C_M_delta_elev_thr_manifold(data, c_m_pitch_fit, c_m_delta_elev_thr_manifold_fit):
    # Plot C_M manifold
    cmap = plt.get_cmap('viridis')
    cnorm = matplotlib.colors.Normalize(10.0,22.5)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    threlevdata = data[
            (abs(data.rig_pitch - 2.5)<0.5)
            & (abs(data.aileron)<2.0)
            & (abs(data.rudder)<2.0)
        ]

    if len(threlevdata.index) == 0:
        return

    c_m_deltas,_ = calculate_c_m_elev(threlevdata,c_m_pitch_fit,None,False)
    
    thing_to_color = ax.scatter(threlevdata.elevator,threlevdata.throttle,c_m_deltas,c=threlevdata.airspeed,norm=cnorm,marker="+")
    
    for aspd in airspeeds:
        elevator_samples = np.linspace(-35,35)
        throttle_samples = np.linspace(0,1)
        T,E = np.meshgrid(throttle_samples,elevator_samples)
        
        c_ne = np.array(c_m_delta_elev_thr_manifold_fit(np.radians(np.ravel(E)),np.ravel(T),aspd))
        CM = c_ne.reshape(E.shape)
        
        ax.plot_surface(E,T,CM,color=[*cmap(cnorm(aspd))[0:3],0.5])

    ax.set_xlabel("Elevator angle (deg)")
    ax.set_ylabel("Throttle setting")
    ax.set_zlabel("C_M contribution")
    fig.colorbar(thing_to_color)

def plot_fit_analysis(data,c_m_pitch_fit,line_fits,surface_fits,manifold_fit):
    fig = plt.figure()
    axs = fig.subplots(2,3)

    colours = ["red","green","blue","orange","purple","brown","yellow","pink","black","grey"]
    positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

    for (p,pitch) in enumerate(pitches):
        for (t,thr) in enumerate(throttles):
            for (i,aspd) in enumerate(airspeeds):
                pointdata = data[
                    (abs(data.rig_pitch - pitch)<0.5)
                    & (abs(data.aileron)<2.0)
                    & (abs(data.rudder)<2.0)
                    & (abs(data.throttle - thr)<0.05)
                    & (abs(data.airspeed - aspd)<0.05)
                    ]
                
                line_fit = None
                if line_fits is not None:
                    line_fit_key = f"c_m_delta_elev@{thr}/{aspd}/{pitch}"
                    if line_fit_key in line_fits:
                        line_fit = line_fits[line_fit_key][0] 
                
                surface_fit = None
                if surface_fits is not None:
                    surface_fit_key = f"c_m_delta_elev@{aspd}"
                    if surface_fit_key in surface_fits:
                        surface_fit = surface_fits[surface_fit_key][0]
                
                (j,k) = positions[i]
                
                samples = np.linspace(-35,35)
                if len(pointdata.index) > 0:
                    c_m_deltas,_ = calculate_c_m_elev(pointdata,c_m_pitch_fit,None,False)
                    axs[j][k].scatter(pointdata.elevator,c_m_deltas,color=colours[t])
                if line_fit is not None:
                    axs[j][k].plot(samples,line_fit(np.radians(samples)),color=colours[t])
                if surface_fit is not None:
                    axs[j][k].plot(samples,surface_fit(np.radians(samples),thr),'.',color=colours[t])
                if manifold_fit is not None:
                    axs[j][k].plot(samples,manifold_fit(np.radians(samples),thr,aspd),'--',color=colours[t])
                axs[j][k].set_title(f"{aspd}")

    plt.show()

# with open("c_m_fits.cpkl","wb") as f:
#     pickle.dump(fits,f)

# import yaml
# with open("c_m_fits.yaml","w") as f:
#     yaml.dump({ k:[float(e) for e in fit[0].args] for (k,fit) in fits.items()},f)

if __name__ == "__main__":
    data = load_data()
    
    elevdata = filter_elev_data(data)
    
    from big3 import get_big3_fits
    _,_,c_m_pitch_fit = get_big3_fits()
    
    c_m_delta_elev_fit = calculate_C_M_delta_elev(data,c_m_pitch_fit)
    plot_C_M_delta_elev(elevdata,c_m_pitch_fit,c_m_delta_elev_fit)
    
    pitches = [2.5]
    
    c_m_delta_elev_thr_fits = calculate_C_M_delta_elev_thr(data,c_m_pitch_fit,c_m_delta_elev_fit)
    plot_C_M_delta_elev_thr(data,c_m_pitch_fit,c_m_delta_elev_thr_fits)

    c_m_delta_elev_thr_surface_fits = calculate_C_M_delta_elev_thr_surfaces(data,c_m_pitch_fit)
    plot_C_M_delta_elev_thr_surfaces(data,c_m_pitch_fit,c_m_delta_elev_thr_surface_fits)

    c_m_delta_elev_thr_manifold_fit,res = calculate_C_M_delta_elev_thr_manifold(data, c_m_pitch_fit)
    print(res)
    plot_C_M_delta_elev_thr_manifold(data, c_m_pitch_fit, c_m_delta_elev_thr_manifold_fit)

    plot_fit_analysis(data,c_m_pitch_fit,c_m_delta_elev_thr_fits,c_m_delta_elev_thr_surface_fits,c_m_delta_elev_thr_manifold_fit)

    plt.show()
