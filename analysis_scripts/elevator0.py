# Do some nonsense to make imports work...
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)

from processing_scripts.utils.fits import Fit

import numpy as np
import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt

def moment_fit_func(elevator, throttle, *args):
    # [m, c] = args
    [m, c] = [ *args, 0]
    return m * throttle * elevator + c


def calculate_moment_fit(data):
    def fit_least_sq(params,data):
        fitted = moment_fit_func(
            np.radians(data.elevator),
            data.throttle,
            *params
        )

        return np.linalg.norm(fitted - data.load_m)

    # x0 = [ 0.037, -0.05 ]
    x0 = [ 0.037 ]
    res = scipy.optimize.minimize(
        fit_least_sq,
        x0,
        args=(data),
        options={"maxiter": 100000}
    )

    return Fit(moment_fit_func,[*res.x]),res


def plot_moment_fit(data, moment_fit):
    cmap = plt.get_cmap('viridis')
    cnorm = matplotlib.colors.Normalize(0,1)

    plt.figure("Moment Fit")
    scatter = plt.scatter(data.elevator, data.load_m, c=data.throttle, norm=cnorm)

    elev_samples = np.arange(-40, 40)
    throttle_samples = np.linspace(0, 1, 10)

    for thr in throttle_samples:
        plt.plot(
            elev_samples,
            moment_fit[0](np.radians(elev_samples), thr),
            c=cmap(cnorm(thr))
        )

    plt.colorbar(scatter)
    plt.xlabel("Elevator Angle (deg)")
    plt.ylabel("Load M (Nm)")


if __name__ == "__main__":
    import pickle

    thisfiledir = os.path.dirname(os.path.abspath(__file__))

    with open(thisfiledir+"/../wind_tunnel_data/processed_corrected/data_11_0.pkl","rb") as f:
        data = pickle.load(f)

    moment_fit = calculate_moment_fit(data)
    print(moment_fit)
    plot_moment_fit(data, moment_fit)

    plt.show()
