import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("c_m_fits.cpkl","rb") as f:
    fits = pickle.load(f)
    
throttles = [0.2,0.35,0.5,0.65,0.8]
airspeeds = [10.0,12.5,15.0,17.5,20.0,22.5]

manifold = fits["c_m_delta_elev"][0]

fig = plt.figure()
axs = fig.subplots(2,3)

colours = ["red","green","blue","orange","purple"]
positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

for (t,thr) in enumerate(throttles):
    for (i,aspd) in enumerate(airspeeds):
        fit = fits[f"c_m_delta_elev@{thr}/{aspd}"][0]
        
        surface = fits[f"c_m_delta_elev@{aspd}"][0]
        
        (j,k) = positions[i]
        
        samples = np.linspace(-30,30)
        axs[j][k].plot(samples,fit(np.radians(samples)),color=colours[t])
        axs[j][k].plot(samples,surface(np.radians(samples),thr),'.',color=colours[t])
        axs[j][k].plot(samples,manifold(np.radians(samples),thr,aspd),'--',color=colours[t])
        axs[j][k].set_title(f"{aspd}")

for fitname in fits:
    fit = fits[fitname]
    print(f"{fit[0].args}: {fit[1]}")

plt.show()
