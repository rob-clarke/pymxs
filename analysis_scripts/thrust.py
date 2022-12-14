# Do some nonsense to make imports work...
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt

from processing_scripts import utils
from processing_scripts.utils.fits import Fit, poly_surface

def calculate_thrust(data,C_lift_fit,C_drag_fit):
    c_lift = C_lift_fit(np.radians(data.pitch))
    c_drag = C_drag_fit(np.radians(data.pitch))

    lift = c_lift * utils.qS(data.airspeed)
    drag = c_drag * utils.qS(data.airspeed)

    # [F_x] = [[ cos(a) -sin(a) ]] [-D]
    # [F_z]   [[ sin(a)  cos(a) ]] [-L]
    expected_load_x = -drag * np.cos(np.radians(data.pitch)) + lift * np.sin(np.radians(data.pitch))
    thrust = data.load_x - expected_load_x
    
    return thrust

def add_zeros(data,thrust):
    # Add artificial zero throttle = zero thrust to data
    example_row = data.iloc[0]
    for airspeed in np.linspace(10,25):
        example_row.throttle = 0
        example_row.airspeed = airspeed
        data = data.append(example_row,ignore_index=True)
        thrust = np.append(thrust,[0])

    return data, thrust    

def thrust_model_surf(thr,aspd,*args):
        # x = thr
        # y = aspd
        
        # z = args[0]*x**2 + args[1]*x*y + args[2]*y**2 + args[3]*x + args[4]*y + args[5]
        
        # #return np.maximum(z,0.0)
        # return z
        return poly_surface(thr,aspd,3,*args)

def calculate_plot_throttle(data,c_lift_fit,c_drag_fit):
    # Calculate C_T
    thrdata = data[
            (abs(data.aileron)<2.0)
            & (data.throttle > 0.2)
            & (abs(data.elevator)<2.0)
            & (abs(data.rudder)<2.0)
            & (abs(data.pitch)<2.0)
            ]

    thrust = calculate_thrust(thrdata, c_lift_fit, c_drag_fit)

    thrdata,thrust = add_zeros(thrdata,thrust)
    thrdata,thrust = add_zeros(thrdata,thrust)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    d = ax.scatter(thrdata.throttle,thrdata.airspeed,thrust)
    # ax = fig.axes[0]
    ax.set_xlabel("Throttle")
    ax.set_ylabel("Airspeed (m/s)")
    ax.set_zlabel("Thrust (N)")
    # cbar = fig.colorbar(d,ax=ax)
    # cbar.set_label("Input power")

    def surface_least_sq(params,data,target_data):
        estimated_data = thrust_model_surf(data.throttle,data.airspeed,*params)
        
        return np.linalg.norm(estimated_data - target_data)

    # import itertools
    # a0 = [1.0,1.0,1.0,1.0]
    # b0 = [10.0,10.0,10.0,10.0]
    # c0 = b0
    # x0 = list(itertools.chain(*zip(c0,b0)))
    x0 = [1.0]*10
    
    bounds = [(-np.inf,np.inf)]*len(x0)
    # Set lower bound on tanh horizontal stretch
    #bounds[3] = (0.3,np.inf)
    
    res = scipy.optimize.minimize(surface_least_sq,x0,args=(thrdata,thrust),method='Nelder-Mead',options={"maxiter":100000},bounds=bounds)
    #print(res)
    
    thrust_model = Fit(thrust_model_surf,res.x)

    throttle_samples = np.linspace(0,0.8)
    airspeed_samples = np.linspace(10,25)
    T,A = np.meshgrid(throttle_samples,airspeed_samples)
    
    fitted_thrust = np.array(thrust_model(np.ravel(T),np.ravel(A)))
    TF = fitted_thrust.reshape(A.shape)
    
    ax.plot_surface(T,A,TF)
    
    print(f"a = {res.x}")
    
    cmap = plt.get_cmap('viridis')
    
    # fig = plt.figure()
    # d = plt.scatter(thrdata.pitch,thrust,c=thrdata.airspeed)
    # ax = fig.axes[0]
    # ax.set_xlabel("Pitch angle (deg)")
    # ax.set_ylabel("Thrust (N)")
    # cbar = fig.colorbar(d,ax=ax)
    # cbar.set_label("Airspeed (m/s)")
    
    cnorm = matplotlib.colors.Normalize(10.0,22.5)
    fig = plt.figure()
    d = plt.scatter(thrdata.throttle,thrdata.rpm/60,c=cmap(cnorm(thrdata.airspeed)))
    ax = fig.axes[0]
    ax.set_xlabel("Throttle")
    ax.set_ylabel("Rev/s")
    cbar = fig.colorbar(d,ax=ax)
    cbar.set_label("Airspeed")
    
    throttles = [0.2,0.35,0.5,0.65,0.8]
    
    cnorm = matplotlib.colors.Normalize(0.0,0.8)
    fig = plt.figure()
    d = plt.scatter(thrdata.airspeed,thrust,c=thrdata.throttle,norm=cnorm)
    ax = fig.axes[0]
    ax.set_xlabel("Airspeed (m/s)")
    ax.set_ylabel("Thrust (N)")
    for t in throttles:
        plt.plot(airspeed_samples,thrust_model(t,airspeed_samples),c=cmap(cnorm(t)))
    cbar = fig.colorbar(d,ax=ax)
    cbar.set_label("Throttle")

    # fig = plt.figure()
    # advance_ratio = thrdata.airspeed / ((thrdata.rpm/60)*utils.prop_rad*2)
    # d = plt.scatter(advance_ratio,thrust,c=thrdata.throttle)
    # for t in throttles:
    #     plt.plot(airspeed_samples / ((100*t+100)*utils.prop_rad*2),thrust_model(t,airspeed_samples))
    # ax = fig.axes[0]
    # ax.set_xlabel("Advance ratio")
    # ax.set_ylabel("Thrust (N)")
    # cbar = fig.colorbar(d,ax=ax)
    # cbar.set_label("Throttle")

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
    from common import load_data
    from big3 import get_big3_fits
    
    c_lift_fit,c_drag_fit,_ = get_big3_fits()
    
    data = load_data()
    
    calculate_plot_throttle(data,c_lift_fit,c_drag_fit)
    plt.show()
    