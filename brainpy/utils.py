import os

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


def path_exists(p):
    return os.path.isdir(p)


def is_valid_output_filename(file_name, exclude_existing=False):
    if os.path.isdir(file_name):
        return False
    elif os.path.isfile(file_name):
        return exclude_existing
    else:
        path = os.path.dirname(file_name)
        return path_exists(path)


def file_exists(file_name):
    return os.path.isfile(file_name)


def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    # convert numpy types such as np.ndarray, np.float64, np.int32, etc to equivalent regular types
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if obj.__class__.__name__ == 'function':
        return 'function %s' % obj.__name__
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return obj
