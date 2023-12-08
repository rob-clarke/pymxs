#!/usr/bin/env python3

import math

from numpy import mean, var

import mxs_data

# Do some more stupid path stuff...
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from analysis_scripts.big3 import c_l_curve, c_d_curve, c_lta, c_dta, x_t, S_w as Sw, S_t, c_w as cw
from processing_scripts.utils.fits import poly_manifold, poly_surface, S

from pyaerso import AffectedBody, AeroBody, Body, Force, Torque


def calc_state(alpha, airspeed, combined=True):
    u = airspeed * math.cos(alpha)
    w = airspeed * math.sin(alpha)
    orientation = [0, math.sin(alpha/2), 0, math.cos(alpha/2)]
    if combined:
        return [
            0, 0, 0,
            u, 0, w,
            *orientation,
            0, 0, 0
        ]
    else:
        return [0, 0, 0], [u, 0, w], orientation, [0, 0, 0]

# Trim points airspeed(m/s): (alpha(deg),elev(deg),throttle)
trim_points = {
    15.0: ( 2.0575192094723063,   -3.8954256604237965, 0.24329511510533755),
    17.5: ( 1.070826277509974,    -4.233529952589323,  0.32366453039697907),
    20.0: ( 0.43018999494832727,  -4.673844770500285,  0.4316004497436157 ),
    22.5: (-0.008982774862057782, -5.099339849609954,  0.559306094219063  ),
    25.0: (-0.32304634892428574,  -5.213030835250511,  0.7004707831407953 )
}


selected_trim_point = 17.5

mass = 1.221
inertia = [
    [0.019, 0.0,   0.0],
    [0.0,   0.09, 0.0],
    [0.0,   0.0,   0.121]]
position,velocity,attitude,rates = calc_state(
    math.radians(trim_points[selected_trim_point][0]),
    selected_trim_point,
    False
)

class WindModel:
    def get_wind(self,position):
        return [0,0,0]
    def step(self,dt):
        pass


class DensityModel:
    def get_density(self,position):
        return 1.225
        # import isacalc
        # [_,_,density,_,_] = isacalc.calculate_at_h(-position[2])
        # return density

def deg2rad(deg):
    return deg/180.0 * math.pi

def clamp(x,l,u):
    return max(min(x,u),l)

c_m_delta_speeds = [
    10.0,
    # 12.5,
    15.0,
    17.5,
    20.0,
    22.5
]
c_m_delta_coeffs = [
    [-1.2437108,  -0.40912014,  0.68492118, -0.41668474, -0.05446331,  1.68333026,
        0.3901467,   1.53671085, -0.03946803,  0.02740918],
    # [-1.16087872, -0.15809144, -1.75358049, -0.59869353, -0.10958262,  3.43234771,
    #     0.6871851,   0.95533266, -0.20000579,  0.05321122],
    [-1.15416664, -0.25519729,  0.44940554, -0.23885613, -0.03351797,  0.51878533,
        0.22735268,  1.51557462, -0.00272833,  0.01202957],
    [-1.14407939, -0.11504951,  0.26004851, -0.51869275,  0.00148918,  0.37951677,
        0.56677261,  1.43193874, -0.10462167,  0.01229135],
    [-0.50449119, -0.05818547,  0.27627828,  0.07019003, -0.11702924,  0.1208649,
        -0.13570306,  1.33357519,  0.09450456,  0.00667436],
    [-0.12412663, -0.01750489, -0.19466328,  0.07393621, -0.07972068,  0.45831077,
        -0.14167883,  1.11579314,  0.08896908,  0.00473207],
    ]

def c_m_surf_get_interp_index(aspd):
    for i,v in enumerate(c_m_delta_speeds):
        if aspd <= v:
            return clamp(i, 1, 5)

class Combined:
    def __init__(self, timestep):
        # self.time = 0
        self.timestep = timestep
        self.alpha_prev = None

    def get_elevator_moment_coeff_delta(self,airstate,rates,input):
        coeffs = [
            -1.09419902e+00,
            -2.11376232e-01,
            2.68838823e-02,
            -2.19078045e+02,
            -1.85268630e-01,
            4.81678617e-03,
            2.19458992e+02,
            -4.51607587e-01,
            2.38686253e-03,
            -1.87506464e-04,
            1.22846953e-03,
            -4.61863302e-01,
            3.63517580e+00,
            5.27810084e-01,
            -1.51973289e-01,
            4.08504874e-03,
            -5.98825396e-02,
            2.59581816e+00,
            -1.73627947e-01,
            9.31570574e-01,
            -4.57208842e+00
        ]
        elev = input[1]
        throttle = input[2]
        aspd = clamp(airstate[2],10,22)
        return poly_manifold(elev,throttle,aspd,3,*coeffs)
    
    def get_elevator_moment_coeff_delta_surf(self, airstate, rates, input):
        elev = input[1]
        throttle = input[2]
        aspd = clamp(airstate[2],10,22.5)
        index = c_m_surf_get_interp_index(aspd)

        lower_aspd = c_m_delta_speeds[index-1]
        lower = poly_surface(elev,throttle,3,*c_m_delta_coeffs[index-1])

        upper_aspd = c_m_delta_speeds[index]
        upper = poly_surface(elev,throttle,3,*c_m_delta_coeffs[index])

        propotion = (aspd - lower_aspd)/(upper_aspd - lower_aspd)
        delta_coeff = upper - lower

        return lower + propotion * delta_coeff

    def get_thrust(self,airspeed,throttle):
        return 1.1576e1 * throttle**2 \
            + -4.8042e-1 * throttle * airspeed \
            + -1.1822e-2 * airspeed**2 \
            + 1.3490e1 * throttle \
            + 4.6518e-1 * airspeed \
            + -4.1090

    def c_l_curve(self,alpha):
        return c_l_curve(alpha, *mxs_data.C_lift_complex)

    def c_d_curve(self,alpha):
        return c_d_curve(alpha, *mxs_data.C_drag_complex)

    def get_cm(self,alpha,c_lw):
        downwash_angle = 2 * c_lw / (math.pi * 4.54)
        alpha = alpha - downwash_angle + math.radians(-0.75)
        tail_lift = c_lta(alpha)
        tail_drag = c_dta(alpha)
        resolved_moment = x_t * S_t * (tail_lift * math.cos(alpha) + tail_drag * math.sin(alpha))
        effective_cm = resolved_moment / (Sw*cw)
        return effective_cm

    def get_m_t_aq(self, c_l, alpha, rates, V, q):
        """
        Get moment generated by the tail due to alpha induced by pitch rate
        """
        downwash_angle = (2 * c_l / (math.pi * 4.54))
        alpha_t = alpha - downwash_angle + math.radians(-0.75)
        alpha_q = math.atan2(
            (-rates[1]*x_t + V*math.sin(alpha_t)),
            (V*math.cos(alpha_t))
        )
        m_t_aq = x_t * q * S_t * (
            math.cos(alpha_q) * c_lta(alpha_q) \
            + math.sin(alpha_q) * c_dta(alpha_q) \
            - math.cos(alpha_t) * c_lta(alpha_t) \
            - math.sin(alpha_t) * c_dta(alpha_t)
        )
        return m_t_aq

    def get_m_t_zero(self, input):
        [_, elev, throttle, _] = input
        return 3.97899352*throttle*elev

    def get_effect(self,airstate,rates,input):
        alpha = airstate[0]
        c_l = self.c_l_curve(alpha)
        c_d = self.c_d_curve(alpha)
        c_m = self.get_cm(alpha,c_l)

        alpha_dot = 0
        if self.alpha_prev is None:
            self.alpha_prev = alpha
        else:
            alpha_dot = (alpha - self.alpha_prev) / self.timestep
            self.alpha_prev = alpha

        q = airstate[3]
        V = airstate[2]

        m_t_aq = self.get_m_t_aq(c_l, alpha, rates, V, q)
        m_t_zero = self.get_m_t_zero(input)

        # dc_m_elev = self.get_elevator_moment_coeff_delta(airstate,rates,input)
        dc_m_elev = self.get_elevator_moment_coeff_delta_surf(airstate,rates,input)

        lift = q * Sw * c_l
        drag = q * Sw * c_d
        # moment = q * Sw * cw * (c_m + dc_m_elev + -6.391 * alpha_dot) + m_t_aq
        moment = q * Sw * cw * (c_m + dc_m_elev)

        if V < 10:
            proportion = V / 10
            moment = (1-proportion) * m_t_zero + proportion * moment

        moment = moment + m_t_aq

        thrust = self.get_thrust(V,input[2])
        # print(f"t: {self.time:.3f} T: {thrust:.3f}, V: {V:.3f}, Alpha: {alpha:.3f}, q: {q:.3f},  c_m: {c_m:.3f}, alpha_q: {alpha_q:.3f}, c_m_eff: {c_m+dc_m_elev:.3f}, moment: {moment:.3f}, dc_m_elev: {dc_m_elev:.3f}, mtaq: {m_t_aq:.3f}")
        # self.time += self.timestep
        x = thrust - drag * math.cos(alpha) + lift * math.sin(alpha)
        z = -lift * math.cos(alpha) - drag * math.sin(alpha)

        # Force and moment fits are found relative to the loadcell point
        # Computed effect forces/moments should be supplied about CG
        #x_cg = 0.03 # m
        #z_cg = -0.016 # m Load cell reference point was 16mm below spar centreline

        x_cg = 0.03 # m
        z_cg = 0 # m
        moment = moment - z * x_cg + x * z_cg

        return (
            Force.body([x,0.0,z]),
            Torque.body([0,moment,0])
            )


def sensible(array,dp=4):
    elements = ", ".join([f"{v:.{dp}f}" for v in array])
    return f"[{elements}]"

def get_alpha_q(alpha,q,V=15):
    result = math.atan2( (-q*x_t + V*math.sin(alpha)), (V*math.cos(alpha)) )
    if result - alpha > 1:
        result = result - 2*math.pi
    return result

def get_cmq(alpha,q,V=15):
    dyn_press = 0.5 * 1.225 * V**2
    alpha_q = get_alpha_q(alpha,q,V)
    m_t_aq = x_t * dyn_press * S_t * (c_lta(alpha_q) - c_lta(alpha))
    return m_t_aq / (Sw * dyn_press * cw * q)

def get_cmqd(alpha, q, V=15):
    dyn_press = 0.5 * 1.225 * V**2
    alpha_q = get_alpha_q(alpha,q,V)
    m_t_aq = x_t * dyn_press * S_t * (
        math.cos(alpha_q) * c_lta(alpha_q) + math.sin(alpha_q) * c_dta(alpha_q)
        - math.cos(alpha) * c_lta(alpha) - math.sin(alpha) * c_dta(alpha)
    )
    return m_t_aq / (Sw * dyn_press * cw * q)

def get_cmqd_dw(alpha, q, V=15):
    dyn_press = 0.5 * 1.225 * V**2

    derate = True

    downwash_angle = 2 * Combined().c_l_curve(alpha) / (math.pi * 4.54) * math.cos(alpha)
    downwash_angle = downwash_angle * ( S(alpha,-math.pi/4,math.pi/4) if derate else 1.0 )
    alpha_t = alpha - downwash_angle + math.radians(-0.5)

    alpha_q = get_alpha_q(alpha_t,q,V)
    m_t_aq = x_t * dyn_press * S_t * (
        math.cos(alpha_q) * c_lta(alpha_q) + math.sin(alpha_q) * c_dta(alpha_q)
        - math.cos(alpha_t) * c_lta(alpha_t) - math.sin(alpha_t) * c_dta(alpha_t)
    )
    return m_t_aq / (Sw * dyn_press * cw * q)

def get_cmqd_dw_vt(alpha, q, V=15):
    V_t = math.sqrt((-q*x_t + V*math.sin(alpha))**2 + (V*math.cos(alpha))**2)
    dyn_press = 0.5 * 1.225 * V**2
    dyn_press_tail = 0.5 * 1.225 * V_t**2

    derate = False

    downwash_angle = 2 * Combined().c_l_curve(alpha) / (math.pi * 4.54) * math.cos(alpha)
    downwash_angle = downwash_angle * ( S(alpha,-math.pi/4,math.pi/4) if derate else 1.0 )
    alpha_t = alpha - downwash_angle + math.radians(-0.5)

    alpha_q = get_alpha_q(alpha_t,q,V)
    m_t_aq = x_t * S_t * (
        dyn_press_tail * (math.cos(alpha_q) * c_lta(alpha_q) + math.sin(alpha_q) * c_dta(alpha_q))
        + dyn_press * (- math.cos(alpha_t) * c_lta(alpha_t) - math.sin(alpha_t) * c_dta(alpha_t))
    )
    return m_t_aq / (Sw * dyn_press * cw * q)

def calc_high_alpha_cmq_eff(q, V=15):
    C_Df = 1.05

    dyn_press = 0.5 * 1.225 * V**2

    V_t = -q*x_t + V
    dyn_press_tail = 0.5 * 1.225 * V_t**2

    m_t = dyn_press_tail * S_t * x_t * C_Df

    m_t_a = dyn_press * S_t * x_t * C_Df

    m_t_aq = m_t - m_t_a

    c_mq_eff = m_t_aq / (Sw * dyn_press * cw * q)

    return c_mq_eff

# print(vehicle.airstate)
if __name__ == "__main__":
    # import sys
    # import numpy as np
    # import matplotlib.pyplot as plt

    # alphas = np.linspace(-180,180,500)

    # # cds = np.zeros_like(alphas)
    # # for i,alpha in enumerate(alphas):
    # #     cds[i] = c_dta(math.radians(alpha))
    # # plt.plot(alphas,cds)

    # qs = [30] #[1, 30, 60] #np.linspace(1,60,3)
    # cmqs = np.zeros((len(qs),len(alphas),4))
    # c_ltas = np.zeros((len(qs),len(alphas),2))
    # alpha_qs = np.zeros((len(qs),len(alphas)))

    # for j,alpha in enumerate(alphas):
    #     for i,q in enumerate(qs):
    #         cmqs[i,j,0] = get_cmq(math.radians(alpha),math.radians(q))
    #         cmqs[i,j,1] = get_cmqd(math.radians(alpha),math.radians(q))
    #         cmqs[i,j,2] = get_cmqd_dw(math.radians(alpha),math.radians(q))
    #         cmqs[i,j,3] = get_cmqd_dw_vt(math.radians(alpha),math.radians(q))
    #         alpha_q = get_alpha_q(math.radians(alpha),math.radians(q))
    #         c_ltas[i,j,0] = c_lta(alpha_q) - c_lta(math.radians(alpha))
    #         c_ltas[i,j,1] = math.cos(alpha_q) * c_lta(alpha_q) + math.sin(alpha_q) * c_dta(alpha_q) \
    #                         - math.cos(math.radians(alpha)) * c_lta(math.radians(alpha)) - math.sin(math.radians(alpha)) * c_dta(math.radians(alpha))
    #         alpha_qs[i,j] = alpha_q

    # static_cmq_effs = [calc_high_alpha_cmq_eff(math.radians(q)) for q in qs]

    # colours = ['k','g','b']

    # for i,q in enumerate(qs):
    #     # plt.plot(alphas,cmqs[i,:,0])
    #     # plt.plot(alphas,cmqs[i,:,1])
    #     plt.plot(alphas,cmqs[i,:,2],f"{colours[i]}:")
    #     plt.plot(alphas,cmqs[i,:,3],f"{colours[i]}-")
    #     plt.plot([-180,180], [static_cmq_effs[i] for _ in [0,0]],f"{colours[i]}--")

    # plt.legend(["Constant dynamic pressure", "Varied dynamic pressure", "Flat plate value"])

    # #plt.legend(list(map(lambda v: str(v),qs)))

    # # ax = plt.gca()
    # # leg = ax.get_legend()
    # # for i,_ in enumerate(qs):
    # #     leg.legendHandles[i].set_color(colours[i])
    # #     leg.legendHandles[i].set_dashes([])
    # plt.xlabel("Angle of attack (deg)")
    # plt.ylabel("$C_{Mq,eff}$")
    # plt.grid(True,'both')

    # plt.figure()
    # for i,q in enumerate(qs):
    #     plt.plot(alphas,np.degrees(alpha_qs[i,:]))
    # plt.legend(list(map(lambda v: str(v),qs)))

    # plt.figure()
    # for i,q in enumerate(qs):
    #     # plt.plot(alphas,c_ltas[i,:,0])
    #     plt.plot(alphas,c_ltas[i,:,1])
    # plt.legend(list(map(lambda v: str(v),qs)))

    # plt.show()

    # sys.exit(0)

    scale = 10
    deltaT = 0.01/scale

    body = Body(mass,inertia,position,velocity,attitude,rates)

    # aerobody = AeroBody(body,WindModel(),("StandardDensity",[]))
    # aerobody = AeroBody(body)
    aerobody = AeroBody(body,None,DensityModel())

    # vehicle = AffectedBody(aerobody,[Lift(),Drag(),Moment()])
    vehicle = AffectedBody(aerobody,[Combined(deltaT)])

    print(sensible(vehicle.statevector))

    import sys
    import time

    outfile = None
    if len(sys.argv) > 1:
        outfile = open(sys.argv[1],"w")
        outfile.write("time,x,y,z,u,v,w,qx,qy,qz,qw,p,q,r,alpha,elevator\n")

    samples = 1
    sample_times = []

    TRIM_ELEVATOR = trim_points[selected_trim_point][1]
    TRIM_THROTTLE = trim_points[selected_trim_point][2]

    def get_doublet_elevator_input(count):
        if count < 500*scale:
            return math.radians(TRIM_ELEVATOR), TRIM_THROTTLE
        if count < 550*scale:
            return math.radians(TRIM_ELEVATOR+5.0), TRIM_THROTTLE
        if count < 600*scale:
            return math.radians(TRIM_ELEVATOR-5.0), TRIM_THROTTLE

        return math.radians(TRIM_ELEVATOR), TRIM_THROTTLE

    def get_loop_input(count):
        throttle = TRIM_THROTTLE
        duration = 600
        if count < 500*scale:
            return math.radians(TRIM_ELEVATOR), throttle
        if count < (500+duration)*scale:
            return math.radians(TRIM_ELEVATOR+15.0), throttle

        if count < 1200*scale:
            return math.radians(TRIM_ELEVATOR), throttle
        if count < (1200+duration-100)*scale:
            return math.radians(TRIM_ELEVATOR+15.0), throttle

        if count < 1900*scale:
            return math.radians(TRIM_ELEVATOR), throttle
        if count < (1900+duration)*scale:
            return math.radians(TRIM_ELEVATOR+15.0), throttle

        return math.radians(TRIM_ELEVATOR), throttle

    def get_elevator_input(count):
        if count < 500*scale:
            return math.radians(TRIM_ELEVATOR), TRIM_THROTTLE
        if count < 600*scale:
            return math.radians(TRIM_ELEVATOR+2.0), TRIM_THROTTLE
        # if count < 700*scale:
        #     return math.radians(TRIM_ELEVATOR), 0.65
        # if count < 1100*scale:
        #     return math.radians(TRIM_ELEVATOR+5.0), 0.65

        return math.radians(TRIM_ELEVATOR), TRIM_THROTTLE

    def constant(level):
        class Constant:
            def __call__(self, t):
                return level
        return Constant

    def pulse(amplitude, t_start, period):
        class Pulse:
            def __call__(self, t):
                if t < t_start:
                    return 0
                if t < (t_start + period):
                    return amplitude
                return 0

    def doublet(amplitude, t_start, period):
        start_pulse = pulse(amplitude, t_start, period/2)
        end_pulse = pulse(amplitude, t_start + period/2, period/2)
        class Doublet:
            def __call__(self, t):
                return start_pulse(t) + end_pulse(t)
        return Doublet

    for i in range(samples):
        count = 0
        simtime = 0

        body = Body(mass,inertia,position,velocity,attitude,rates)
        # aerobody = AeroBody(body,None,DensityModel())
        aerobody = AeroBody(body,None,("StandardDensity",[]))
        vehicle = AffectedBody(aerobody,[Combined(deltaT)])

        start = time.process_time()
        while count < 3000*scale:
            elevator, throttle = get_doublet_elevator_input(count)
            # elevator, throttle = get_loop_input(count)
            # elevator, throttle = get_elevator_input(count)
            vehicle.step(deltaT,[0,elevator,throttle])
            count += 1
            simtime += deltaT
            vehicle.statevector
            if outfile:
                outfile.write(f"{simtime},"+sensible(vehicle.statevector,10)[1:-1] + f",{vehicle.airstate[0]},{elevator}\n")
        end = time.process_time()
        sample_times.append(end-start)
        print(sensible(vehicle.statevector))

    print(f"Mean: {mean(sample_times)}\nVar: {var(sample_times)}")
    if outfile is not None:
        outfile.close()
