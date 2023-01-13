import math

from analysis_scripts.big3 import c_l_curve
import models.mxs_data as mxs_data

ar = 4.54 # Aspect ratio
c_s = 0.19 # Tip chord
c_r = 0.29 # Root chord
lam = c_s / c_r # Taper ratio
xt = 0.585 # Tail offset (x)
zt = 0.035 # Tail offset (z) 
b = 1.09 # Span
c = 0.24 # MAC
c_t = 0.165 # Tail MAC
sweep = 0
S = 0.263 # Wing area
St = 0.0825 # Tail area

alpha = math.radians(5)

C_L0 = 0.1476
C_L = c_l_curve(alpha, *mxs_data.C_lift_complex)
C_D0 = 0.0671
C_LaT = 3.581

print(F"{C_L=}")

lh = xt - 0.25*c_r + 0.25*c_t
hh = zt

ka = 1/ar - 1/(1 + math.pow(ar,1.7))
kl = (10 - 3*lam)/7
kh = (1-hh/b) / math.pow(2*lh/b, 1/3)

deda = 4.44 * math.pow(
  ka * kl * kh * math.sqrt(math.cos(sweep)),
  1.19
)

print(f"{ka=}\n{kl=}\n{kh=}\n{deda=}")

x_tte = xt - c_r + 0.25*c_t

zw_c = 0.68 * math.sqrt(C_D0 * (x_tte/c + 0.15))
print(f"{zw_c=}")

downwash_ang = 1.62 * C_L / (math.pi * ar)
print(f"{downwash_ang=}")

# gamma = angle off centreline from wing TE to tail quarter chord
gamma = math.atan(zt / (xt - c_r))

z_vort = x_tte * math.tan(gamma + downwash_ang - alpha)
print(f"{z_vort=}")

dqq0 = 2.42 * math.sqrt(C_D0) / (x_tte/c + 0.3)
print(f"{dqq0=}")

dqq = dqq0 * math.pow(
  math.cos(math.pi/2 * z_vort / (zw_c * c)),
  2
)
print(f"{dqq=}")

etah = 1 - dqq
print(f"{etah=}")

tail_vol = xt * St / (c * S)
cmadot = -2 * C_LaT * etah * tail_vol * xt / c * deda

print(f"{tail_vol=}")
print(f"{cmadot=}")
