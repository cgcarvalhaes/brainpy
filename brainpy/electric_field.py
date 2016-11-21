import numpy as np
from laplacian import Laplacian
from numpy.polynomial import legendre
from .utils import cart_to_sph


class ElectricField(Laplacian):
    def __init__(self, locations, m, truncation=100):
        super(ElectricField, self).__init__(locations, m, truncation=truncation)

        # the derivative of cos_gamma with respect to theta
        x, y, z = self.normalized_locations[:, 0], self.normalized_locations[:, 1], self.normalized_locations[:, 2]
        r, theta, phi = cart_to_sph(x, y, z)
        locations_theta = np.c_[np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
        cos_gamma_diff_theta = np.dot(locations_theta, self.normalized_locations.T)

        # the derivative of cos_gamma with respect to phi
        locations_phi = np.c_[-np.sin(phi), np.cos(phi), np.zeros_like(phi)]
        cos_gamma_diff_phi = np.dot(locations_phi, self.normalized_locations.T)

        c_legendre_der = legendre.legder(self.c_legendre)
        leg_val_der = legendre.legval(self.cos_gamma, c_legendre_der)
        self.mat_gm_theta = (0.25/np.pi) * cos_gamma_diff_theta * leg_val_der
        self.mat_gm_phi = (0.25/np.pi) * cos_gamma_diff_phi * leg_val_der

        self.theta_mat = None
        self.phi_mat = None

    def eval(self, lambda_value=0):
        super(ElectricField, self).eval(lambda_value)
        self.theta_mat = np.dot(self.mat_gm_theta, self.C)
        self.phi_mat = np.dot(self.mat_gm_phi, self.C)
        return self

    def transform(self, data):
        self.assert_data_shape_is_alright(data)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        data = np.c_[data, data, data]
        data[:, :, 0] = super(ElectricField, self).transform(data[:, :, 0].squeeze())
        data[:, :, 1] = np.dot(self.theta_mat, data[:, :, 1].squeeze())
        data[:, :, 2] = np.dot(self.phi_mat, data[:, :, 2].squeeze())
        return data
