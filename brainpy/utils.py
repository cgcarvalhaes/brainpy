import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d


def norm_rows(v):
    r = np.linalg.norm(v, axis=1)
    u = np.divide(v, r[:, None])
    return u


def cart_to_sph(x, y, z):
    hypot_xy = np.hypot(x, y)
    r = np.hypot(hypot_xy, z)
    theta = np.arctan2(hypot_xy, z)
    phi = np.arctan2(y, x)
    return r, theta, phi


def find_minimum(x, y):
    # down sample x to make sure its elements are unique within a tolerance
    x_ds, j = np.unique(x.round(0), return_index=True)
    y_ds = y[j]
    k = x_ds.argsort()
    cubic_splines = interp1d(x_ds[k], y_ds[k], kind='cubic')
    df_opt, e_min, ierr, num_func = optimize.fminbound(cubic_splines, x_ds.min(), x_ds.max(), full_output=True)
    return df_opt, e_min, ierr, num_func, x_ds, y_ds


def cubic_splines_interpolation(x, y, x0):
    x = np.asarray(x)
    y = np.asarray(y)
    k = x.argsort()
    cubic_splines = interp1d(x[k], y[k], kind='cubic')
    return cubic_splines(x0)
