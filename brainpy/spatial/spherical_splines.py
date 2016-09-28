import numpy as np
from numpy.polynomial import legendre

from spatial.utils import norm_rows


class SphericalSplines(object):
    def __init__(self, locations, m, truncation=100):
        locations = np.asarray(locations)

        assert np.ndim(locations) == 2, 'The locations must be a two-dimensional array of shape N by 3'
        assert m in [3, 4, 5], 'The parameter "m" must be either 3, 4 or 5'

        self.m = m

        # normalize locations to the unit sphere
        self.normalized_locations = norm_rows(locations)

        # compute the scalar product of each pair of locations
        self.cos_gamma = np.dot(self.normalized_locations, self.normalized_locations.T)

        # the matrix gm of the spherical splines expansion
        self.ell = np.arange(1, truncation + 1)
        self.c_legendre = (2. * self.ell + 1.) / (np.power(self.ell, m) * np.power(self.ell + 1., m))
        self.G = (0.25 / np.pi) * legendre.legval(self.cos_gamma, self.c_legendre)

        # QR-decomposition ot the T matrix
        self.T = np.ones((len(self.G), 1))
        self.Q, self.r_mat = np.linalg.qr(self.T, mode='complete')
        self.Q1 = self.Q[:, 0][:, None]
        self.Q2 = self.Q[:, 1:]
        self.R = self.r_mat[0, 0]
        self.n = len(self.T)
        self.Id = np.eye(self.n)

        self.C = None
        self.D = None
        self.smoother = None

    def eval(self, lambda_value=0):
        # solve the splines linear system
        GI = self.G + self.n * lambda_value * self.Id
        self.C = reduce(np.dot, [self.Q2, np.linalg.inv(reduce(np.dot, [self.Q2.T, GI, self.Q2])), self.Q2.T])
        self.D = np.dot(self.Q1.T, (self.Id - np.dot(GI, self.C))) / self.R
        self.smoother = np.dot(self.G, self.C) + np.dot(self.T, self.D)
        return self

    def transform(self, data):
        return np.dot(self.smoother, data)
