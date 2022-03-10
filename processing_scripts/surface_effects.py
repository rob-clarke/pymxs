from cmath import inf
import pickle
import matplotlib
import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt

from . import utils

from .utils.fits import linear, quadratic, tanh, H, S, P, Fit

chord = utils.params.chord

def add_airspeed_column(df,airspeed):
    airspeed = [airspeed]*len(df.index)
    df['airspeed'] = airspeed

def c_l_curve(alpha,cl_0,cl_alpha,pstall,nstall):
    #astall = np.radians(10)
    return S(alpha,nstall,pstall) * (cl_0 + cl_alpha * alpha) # Linear regime

def calculate_C_lift(neutral_data):
    # Calculate C_lift
    
    c_lifts = neutral_data.lift / utils.qS(neutral_data.airspeed)

    popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_lifts)
    C_lift = {
        "zero": float(popt[1]),
        "alpha": float(popt[0])
        }
    popt,pcov = scipy.optimize.curve_fit(c_l_curve,np.radians(neutral_data.pitch),c_lifts,[C_lift["zero"],C_lift["alpha"],np.radians(10),-np.radians(10)])
    C_lift_complex = popt
    
    return c_lifts,C_lift,C_lift_complex


def c_d_curve(alpha,cd_0,cd_alpha,alpha_cd0):
    alpha_lim = 30
    return S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim)) * (cd_alpha*(alpha-alpha_cd0)**2 + cd_0) \
        + (1.0-S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim))) * 2.0

def calculate_C_drag(neutral_data):
    # Calculate C_drag
        
    c_ds = neutral_data.drag / utils.qS(neutral_data.airspeed)

    popt,pcov = scipy.optimize.curve_fit(quadratic,np.radians(neutral_data.pitch),c_ds)
    C_drag = {
        "zero": float(popt[2]),
        "alpha": float(popt[1]),
        "alpha2": float(popt[0])
        }

    popt,pcov = scipy.optimize.curve_fit(c_d_curve,np.radians(neutral_data.pitch),c_ds,maxfev=10000)
    C_drag_complex = popt
    
    return c_ds,C_drag,C_drag_complex


def c_m_curve(alpha,cm_0,alpha_cm0,cm_hscale,cm_vscale):
    alpha_lim = 15
    asymptote = 0.5
    k=12
    return S(alpha,np.radians(-alpha_lim),np.radians(alpha_lim),k) * (cm_vscale*np.tan(cm_hscale*(alpha-alpha_cm0)) + cm_0) \
        + (1.0-H(alpha,np.radians(-alpha_lim),k)) * asymptote \
        + H(alpha,np.radians(alpha_lim),k) * -asymptote

def calculate_C_M(neutral_data):
    # Calculate C_M
        
    c_ms = neutral_data.load_m / utils.qSc(neutral_data.airspeed)

    popt,pcov = scipy.optimize.curve_fit(linear,np.radians(neutral_data.pitch),c_ms)
    C_M = {
        "zero": float(popt[1]),
        "alpha": float(popt[0])
        }

    popt,pcov = scipy.optimize.curve_fit(c_m_curve,np.radians(neutral_data.pitch),c_ms,maxfev=10000)
    C_M_complex = popt
    
    return c_ms,C_M,C_M_complex

def make_plot(ax,xs,ys,samples1,fit1,samples2,fit2,xlabel,ylabel,grid=True):
    ax.scatter(xs,ys)
    ax.plot(samples1,fit1(samples1),c="red")
    if fit2 is not None:
        ax.plot(samples2,fit2(samples2),c="green")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid,'both')

def make_top3_plots(neutral_data,c_lifts,C_lift,C_lift_complex,c_ds,C_drag,C_drag_complex,c_ms,C_M,C_M_complex):
    # Make plots
    #fig,axs = plt.subplots(1,3)

    # Plot lift
    fig = plt.figure()
    ax = plt.axes()
    make_plot(ax,
        neutral_data.pitch,c_lifts,
        np.linspace(-15,15),
        lambda x: linear(np.radians(x),C_lift["alpha"],C_lift["zero"]),
        np.linspace(-30,30,200),
        lambda x: c_l_curve(np.radians(x),*C_lift_complex),
        "Pitch angle (deg)",
        "C_L"
        )

    # Plot drag
    fig = plt.figure()
    ax = plt.axes()
    make_plot(ax,
        neutral_data.pitch,c_ds,
        np.linspace(-15,15),
        lambda x: quadratic(np.radians(x),C_drag["alpha2"],C_drag["alpha"],C_drag["zero"]),
        np.linspace(-30,30,200),
        lambda x: c_d_curve(np.radians(x),*C_drag_complex),
        "Pitch angle (deg)",
        "C_D"
        )

    # Plot C_M
    fig = plt.figure()
    ax = plt.axes()
    make_plot(ax,
        neutral_data.pitch,c_ms,
        np.linspace(-15,15),
        lambda x: linear(np.radians(x),C_M["alpha"],C_M["zero"]),
        np.linspace(-30,30,200),
        lambda x: c_m_curve(np.radians(x),*C_M_complex),
        "Pitch angle (deg)",
        "C_M"
        )



def calculate_C_M_elev(elevdata,C_M):
    # C_M_elev
    c_m_alpha = linear(np.radians(elevdata.pitch),C_M["alpha"],C_M["zero"])
    c_ms = elevdata.load_m / utils.qSc(elevdata.airspeed) - c_m_alpha
    popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(elevdata.elevator),c_ms)
    C_Me = {
        "elev_a": float(popt[0]),
        "elev_b": float(popt[1]),
        "elev_c": float(popt[2]),
        "elev_d": float(popt[3]),
        }
    return c_ms,C_Me


def calculate_C_L_ail(aildata):
    # Calculate C_L_ail

    c_ls = aildata.load_l / utils.qSc(aildata.airspeed)
    popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(aildata.aileron),c_ls)
    C_La = {
        "ail_a": float(popt[0]),
        "ail_b": float(popt[1]),
        "ail_c": float(popt[2]),
        "ail_d": float(popt[3])
        }
    
    return c_ls,C_La

def calculate_C_L_rudd(rudddata):
    # Calculate C_L_rudd

    c_ls = rudddata.load_l / utils.qSc(rudddata.airspeed)
    popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(rudddata.rudder),c_ls)
    C_Lr = {
        "rudd_a": float(popt[0]),
        "rudd_b": float(popt[1]),
        "rudd_c": float(popt[2]),
        "rudd_d": float(popt[3]),
    }

    return c_ls,C_Lr


def calculate_C_N_rudd(rudddata):
    # Calculate C_N_rudd

    c_ns = rudddata.load_n / utils.qSc(rudddata.airspeed)
    popt,pcov = scipy.optimize.curve_fit(tanh,np.radians(rudddata.rudder),c_ns)
    C_Nr = {
        "rudd_a": float(popt[0]),
        "rudd_b": float(popt[1]),
        "rudd_c": float(popt[2]),
        "rudd_d": float(popt[3])
        }
    
    return c_ns,C_Nr

def make_control_plots(elevdata,aildata,rudddata,c_ms,C_Me,c_lsa,C_La,c_lsr,C_Lr,c_nsr,C_Nr):
    fig,axs = plt.subplots(2,3)

    axs[0,0].scatter(elevdata.elevator,c_ms,c=elevdata.pitch)
    fit_samples = np.linspace(-30,30)
    fitted = tanh(np.radians(fit_samples),C_Me["elev_a"],C_Me["elev_b"],C_Me["elev_c"],C_Me["elev_d"])
    axs[0,0].scatter(fit_samples,fitted,c="red")
    axs[0,0].set_xlabel("Elevator angle (deg)")
    axs[0,0].set_ylabel("C_M contribution")

    axs[0,1].scatter(aildata.aileron,c_lsa,c=aildata.pitch)
    fit_samples = np.linspace(-30,30)
    fitted = tanh(np.radians(fit_samples),C_La["ail_a"],C_La["ail_b"],C_La["ail_c"],C_La["ail_d"])
    axs[0,1].scatter(fit_samples,fitted,c="red")
    axs[0,1].set_xlabel("Aileron angle (deg)")
    axs[0,1].set_ylabel("C_L contribution")

    axs[1,1].scatter(rudddata.rudder,c_lsr,c=rudddata.pitch)
    fit_samples = np.linspace(-30,30)
    fitted = tanh(np.radians(fit_samples),C_Lr["rudd_a"],C_Lr["rudd_b"],C_Lr["rudd_c"],C_Lr["rudd_d"])
    axs[1,1].scatter(fit_samples,fitted,c="red")
    axs[1,1].set_xlabel("Rudder angle (deg)")
    axs[1,1].set_ylabel("C_L contribution")

    axs[0,2].scatter(rudddata.rudder,c_nsr,c=rudddata.pitch)
    fit_samples = np.linspace(-30,30)
    fitted = tanh(np.radians(fit_samples),C_Nr["rudd_a"],C_Nr["rudd_b"],C_Nr["rudd_c"],C_Nr["rudd_d"])
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

def plot_c_n_rudd(data):
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


# # Calc C_N surfaces
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# def c_n_surf(rudd,thr,am,ac,bm,bc,cm,cc,dm,dc):
#     a = linear(thr,am,ac)
#     b = linear(thr,bm,bc)
#     c = linear(thr,cm,cc)
#     d = linear(thr,dm,dc)
#     return tanh(rudd,a,b,c,d)

# def surface_least_sq(params,data,c_ns):
#     c_ne = c_n_surf(np.radians(data.rudder),data.throttle,*params)
    
#     return np.linalg.norm(c_ne - c_ns)

# import itertools

# for aspd in airspeeds:
#     thrrudddata = data[
#             (abs(data.rig_pitch - 2.5)<0.5)
#             & (abs(data.aileron)<2.0)
#             & (abs(data.elevator)<2.0)
#             & (abs(data.airspeed - aspd)<0.05)
#         ]

#     if len(thrrudddata.index) != 0:
#         c_ns,_ = calculate_c_n_rudd(thrrudddata,False)
#         m0 = [0.0,0.0,0.0,0.0]
#         c0 = [-2.9999798, 0.45542569, 0.4081167, 1.08740676]
#         x0 = list(itertools.chain(*zip(m0,c0)))
#         res = scipy.optimize.minimize(surface_least_sq,x0,args=(thrrudddata,c_ns),method='Powell',options={"maxiter":100000})
#         #print(res)
#         print(f"Optimised for aspd:{aspd:5.1f}: {res.x}")
        
#         thing_to_color = ax.scatter(thrrudddata.rudder,thrrudddata.throttle,c_ns,c=thrrudddata.airspeed,norm=cnorm,marker="+")
        
#         rudder_samples = np.linspace(-30,30)
#         throttle_samples = np.linspace(0,1)
#         T,R = np.meshgrid(throttle_samples,rudder_samples)
        
#         c_ne = np.array(c_n_surf(np.radians(np.ravel(R)),np.ravel(T),*res.x))
#         CN = c_ne.reshape(R.shape)
        
#         ax.plot_surface(R,T,CN,color=[*cmap(cnorm(aspd))[0:3],0.5])

# ax.set_xlabel("Rudder angle (deg)")
# ax.set_ylabel("Throttle setting")
# ax.set_zlabel("C_N contribution")
# #fig.colorbar(thing_to_color)

# # Calc C_N manifold
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# def c_n_manif(rudd,thr,aspd,amm,amc,acm,acc,bmm,bmc,bcm,bcc,cmm,cmc,ccm,ccc,dmm,dmc,dcm,dcc):
#     am = linear(aspd,amm,amc)
#     ac = linear(aspd,acm,acc)
#     bm = linear(aspd,bmm,bmc)
#     bc = linear(aspd,bcm,bcc)
#     cm = linear(aspd,cmm,cmc)
#     cc = linear(aspd,ccm,ccc)
#     dm = linear(aspd,dmm,dmc)
#     dc = linear(aspd,dcm,dcc)
#     return c_n_surf(rudd,thr,am,ac,bm,bc,cm,cc,dm,dc)

# def manif_least_sq(params,data,c_ns):
#     c_ne = c_n_manif(np.radians(data.rudder),data.throttle,data.airspeed,*params)
    
#     return np.linalg.norm(c_ne - c_ns)

# thrrudddata = data[
#         (abs(data.rig_pitch - 2.5)<0.5)
#         & (abs(data.aileron)<2.0)
#         & (abs(data.elevator)<2.0)
#     ]

# if len(thrrudddata.index) != 0:
#     c_ns,_ = calculate_c_n_rudd(thrrudddata,False)
#     c0 = [-0.13642334, -2.8655547, 0.28847784, 0.26922272, 0.00860537, 0.40901526, -0.01916245, 1.10104976]
#     m0 = [0.0]*len(c0)
#     x0 = list(itertools.chain(*zip(m0,c0)))
#     res = scipy.optimize.minimize(manif_least_sq,x0,args=(thrrudddata,c_ns),method='Powell',options={"maxiter":100000})
#     # print(res)
#     print(f"Optimised for aspd:{aspd:5.1f}: {res.x}")
    
#     thing_to_color = ax.scatter(thrrudddata.rudder,thrrudddata.throttle,c_ns,c=thrrudddata.airspeed,norm=cnorm,marker="+")
    
#     for aspd in airspeeds:
#         rudder_samples = np.linspace(-30,30)
#         throttle_samples = np.linspace(0,1)
#         T,R = np.meshgrid(throttle_samples,rudder_samples)
        
#         c_ne = np.array(c_n_manif(np.radians(np.ravel(R)),np.ravel(T),aspd,*res.x))
#         CN = c_ne.reshape(R.shape)
        
#         ax.plot_surface(R,T,CN,color=[*cmap(cnorm(aspd))[0:3],0.5])

# ax.set_xlabel("Rudder angle (deg)")
# ax.set_ylabel("Throttle setting")
# ax.set_zlabel("C_N contribution")
# #fig.colorbar(thing_to_color)

def calculate_plot_throttle(data,C_lift_complex,C_drag_complex):
    # Calculate C_T
    thrdata = data[
            (abs(data.aileron)<2.0)
            & (data.throttle > 0.1)
            & (abs(data.elevator)<2.0)
            & (abs(data.rudder)<2.0)
            & (abs(data.pitch)<2.0)
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

    # # Add artificial zeros
    # print(f"Lengths before: {len(thrdata.index)}, {len(thrust)}")
    # example_row = thrdata.iloc[0]
    # for airspeed in np.linspace(10,25):
    #     example_row.throttle = 0
    #     example_row.airspeed = airspeed
    #     thrdata = thrdata.append(example_row,ignore_index=True)
    #     thrust = np.append(thrust,[0])

    # print(f"Lengths after: {len(thrdata.index)}, {len(thrust)}")
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    d = ax.scatter(thrdata.throttle,thrdata.airspeed,thrust)
    # ax = fig.axes[0]
    ax.set_xlabel("Throttle")
    ax.set_ylabel("Airspeed (m/s)")
    ax.set_zlabel("Thrust (N)")
    # cbar = fig.colorbar(d,ax=ax)
    # cbar.set_label("Input power")


    def thrust_surf(thr,aspd,*args):
        # if len(args) % 4 != 0:
        #     raise ValueError("length of args must be divisible by 3")
        # poly_order_p1 = len(args) // 3
        # a = P(thr,*args[0*poly_order_p1:1*poly_order_p1])
        # b = P(thr,*args[1*poly_order_p1:2*poly_order_p1])
        # c = P(thr,*args[2*poly_order_p1:3*poly_order_p1])
        # return quadratic(aspd,a,b,c)
        x = thr
        y = aspd
        z = args[0]*x**2 + args[1]*y**2 + args[2]*x*y + args[3]*x + args[4]*y + args[5]
        return z

    def surface_least_sq(params,data,target_data):
        estimated_data = thrust_surf(data.throttle,data.airspeed,*params)
        
        return np.linalg.norm(estimated_data - target_data)

    # import itertools
    # a0 = [1.0,1.0,1.0,1.0]
    # b0 = [10.0,10.0,10.0,10.0]
    # c0 = b0
    # x0 = list(itertools.chain(*zip(c0,b0)))
    x0 = [1.0]*6
    
    bounds = [(-np.inf,np.inf)]*len(x0)
    # Set lower bound on tanh horizontal stretch
    #bounds[3] = (0.3,np.inf)
    
    res = scipy.optimize.minimize(surface_least_sq,x0,args=(thrdata,thrust),method='Nelder-Mead',options={"maxiter":100000},bounds=bounds)
    #print(res)
    
    thrust_model = Fit(thrust_surf,res.x)

    throttle_samples = np.linspace(0,0.8)
    airspeed_samples = np.linspace(10,25)
    T,A = np.meshgrid(throttle_samples,airspeed_samples)
    
    fitted_thrust = np.array(thrust_model(np.ravel(T),np.ravel(A)))
    TF = fitted_thrust.reshape(A.shape)
    
    ax.plot_surface(T,A,TF)
    
    print(f"a = {res.x}")
    
    # fig = plt.figure()
    # d = plt.scatter(thrdata.pitch,thrust,c=thrdata.airspeed)
    # ax = fig.axes[0]
    # ax.set_xlabel("Pitch angle (deg)")
    # ax.set_ylabel("Thrust (N)")
    # cbar = fig.colorbar(d,ax=ax)
    # cbar.set_label("Airspeed (m/s)")
    
    fig = plt.figure()
    d = plt.scatter(thrdata.throttle,thrdata.rpm/60,c=thrdata.airspeed)
    ax = fig.axes[0]
    ax.set_xlabel("Throttle")
    ax.set_ylabel("Rev/s")
    cbar = fig.colorbar(d,ax=ax)
    cbar.set_label("Airspeed")
    
    throttles = [0.2,0.35,0.5,0.65,0.8]
    
    fig = plt.figure()
    d = plt.scatter(thrdata.airspeed,thrust,c=thrdata.throttle)
    ax = fig.axes[0]
    ax.set_xlabel("Airspeed (m/s)")
    ax.set_ylabel("Thrust (N)")
    for t in throttles:
        plt.plot(airspeed_samples,thrust_model(t,airspeed_samples))
    cbar = fig.colorbar(d,ax=ax)
    cbar.set_label("Throttle")

    fig = plt.figure()
    advance_ratio = thrdata.airspeed / ((thrdata.rpm/60)*utils.prop_rad*2)
    d = plt.scatter(advance_ratio,thrust,c=thrdata.throttle)
    for t in throttles:
        plt.plot(airspeed_samples / ((100*t+100)*utils.prop_rad*2),thrust_model(t,airspeed_samples))
    ax = fig.axes[0]
    ax.set_xlabel("Advance ratio")
    ax.set_ylabel("Thrust (N)")
    cbar = fig.colorbar(d,ax=ax)
    cbar.set_label("Throttle")

    # fig = plt.figure()
    # prop_power = thrust * thrdata.airspeed
    # elec_power = thrdata.current*thrdata.voltage
    # d = plt.scatter(advance_ratio,prop_power,c=thrdata.throttle)
    # d = plt.scatter(advance_ratio,elec_power,marker="+",c="grey")
    # ax = fig.axes[0]
    # ax.set_xlabel("Advance ratio")
    # ax.set_ylabel("Power (W)")
    # ax.legend(("Propeller power", "Electrical power"))
    # cbar = fig.colorbar(d,ax=ax)
    # cbar.set_label("Throttle")

    # fig = plt.figure()
    # d = plt.scatter(advance_ratio,prop_power/elec_power,c=thrdata.throttle)
    # ax = fig.axes[0]
    # ax.set_xlabel("Advance ratio")
    # ax.set_ylabel("Prop efficiency")
    # cbar = fig.colorbar(d,ax=ax)
    # cbar.set_label("Throttle")
    # # plt.axis([0, 0.55, 0, 1])

    # fig = plt.figure()
    # d = plt.scatter(thrdata.rpm/60,thrust,c=thrdata.airspeed)
    # ax = fig.axes[0]
    # ax.set_xlabel("Rev/s")
    # ax.set_ylabel("Thrust")
    # cbar = fig.colorbar(d,ax=ax)
    # cbar.set_label("Input power")


    # fig = plt.figure()
    # ax = plt.axes()
    # make_plot(ax,
    #     thrdata.rpm/60,thrust,
    #     np.linspace(100,200),
    #     lambda x: np.zeros(x.size),
    #     None,
    #     None,
    #     "Rev/s",
    #     "Thrust")
        

    # rho = 1.225
    # D = 0.28
    # n = thrdata.rpm / 60.0

    # k_t = thrust / (rho * n**2 * D**4)

    # k_t_select = (thrust > 1.0) & (thrdata.rpm > 100)

    # d = axs[1].scatter(thrdata[k_t_select].pitch,k_t[k_t_select],c=thrdata[k_t_select].throttle)
    # axs[1].set_xlabel("Pitch angle (deg)")
    # axs[1].set_ylabel("Thrust coefficient")
    # cbar = fig.colorbar(d,ax=axs[1])
    # cbar.set_label("Throttle setting")

# import yaml
# print(yaml.dump({
#     "C_lift": C_lift,
#     "C_drag": C_drag,
#     "C_L": C_L,
#     "C_M": C_M,
#     "C_N": C_N,
#     }))

# plt.show()

if __name__ == "__main__":
    import os
    thisfiledir = os.path.dirname(os.path.abspath(__file__))
    
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
        newdata = pickle.load(open(thisfiledir+"/../wind_tunnel_data/processed/"+filename,"rb"))
        add_airspeed_column(newdata,airspeed)
        data = pd.concat([data,newdata])

    neutral_data = utils.select_neutral(data)
    
    c_lifts,C_lift,C_lift_complex = calculate_C_lift(neutral_data)
    c_ds,C_drag,C_drag_complex = calculate_C_drag(neutral_data)
    c_ms,C_M,C_M_complex = calculate_C_M(neutral_data)
    
    make_top3_plots(neutral_data,
        c_lifts,C_lift,C_lift_complex,
        c_ds,C_drag,C_drag_complex,
        c_ms,C_M,C_M_complex
        )
        
    elevdata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.rudder)<2.0)
        ]

    aildata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.elevator)<2.0)
        & (abs(data.rudder)<2.0)
        ]

    rudddata = data[
        (abs(data.throttle)<0.05)
        & (abs(data.aileron)<2.0)
        & (abs(data.elevator)<2.0)
        ]
    
    calculate_plot_throttle(data,C_lift_complex,C_drag_complex)
    plt.show()
    