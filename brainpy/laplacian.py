import numpy as np
from numpy.polynomial import legendre

from .spherical_splines import SphericalSplines


class Laplacian(SphericalSplines):
    def __init__(self, locations, m, truncation=100):
        super(Laplacian, self).__init__(locations, m, truncation=truncation)
        c_legendre_lap = -(2. * self.ell + 1.) / (self.ell**(self.m - 1) * (self.ell + 1.)**(self.m - 1))
        self.mat_gm_lap = (0.25/np.pi) * legendre.legval(self.cos_gamma, c_legendre_lap)
        self.lap_mat = None

    def eval(self, lambda_value=0):
        super(Laplacian, self).eval(lambda_value)
        self.lap_mat = np.dot(self.mat_gm_lap, self.C)
        return self

    def transform(self, data):
        return np.dot(self.lap_mat, np.asarray(data))
