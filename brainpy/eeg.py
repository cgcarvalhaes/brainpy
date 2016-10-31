import numpy as np
from etc import group_labels
from laplacian import Laplacian
from utils import file_exists

from .electric_field import ElectricField


class EEG(object):
    DEFAULT_SAMPLING_RATE = 1e3

    def __init__(self, data=None, trial_size=None, electrodes=None, subject=None, sampling_rate=DEFAULT_SAMPLING_RATE,
                 trial_labels=None, data_reader=None, is_laplacian=False, is_electric_field=False, filename=None):
        self.trial_size = trial_size
        self.data = data
        if self.data is None:
            self.data = np.array([])
        self._electrodes = electrodes
        if self._electrodes is None:
            self._electrodes = dict()
        self.subject = subject
        self.sampling_rate = sampling_rate
        self._trial_labels = trial_labels
        if self._trial_labels is None:
            self._trial_labels = np.array([])
        self.data_reader = data_reader
        self._cache = dict()
        self._is_laplacian = is_laplacian
        self._is_electric_field = is_electric_field
        self.filename = filename
        self.group_size = 1

    def get_params(self):
        return {
            "trial_size": self.trial_size,
            "data_shape": self.data.shape,
            "electrodes": self.electrodes,
            "subject": self.subject,
            "sampling_rate": self.sampling_rate,
            "trial_labels": self.trial_labels,
            "is_laplacian": self.is_laplacian,
            "is_electric_field": self.is_electric_field,
            "filename": self.filename,
            "group_size": self.group_size,
            "n_channels": self.n_channels,
            "n_trials": self.n_trials,
            "n_comps": self.n_comps,
        }

    @property
    def trial_labels(self):
        return np.asarray(self._trial_labels)

    @property
    def electrodes(self):
        return list(self._electrodes)

    @property
    def n_channels(self):
        return self.data.shape[0]

    @property
    def n_samples(self):
        if self.data.ndim > 1:
            return self.data.shape[1]
        return 0

    @property
    def n_trials(self):
        if self.trial_size:
            return self.n_samples // self.trial_size
        return 1

    @property
    def n_comps(self):
        return 0 if self.data.ndim == 2 else self.data.shape[2]

    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def has_trials(self):
        return self.n_trials > 1

    @property
    def has_electrodes(self):
        return len(self.electrodes) > 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def is_potential(self):
        return not self._is_laplacian and not self._is_electric_field

    @property
    def is_laplacian(self):
        return self._is_laplacian

    @property
    def is_electric_field(self):
        return self._is_electric_field

    def _check_valid_data_format(self, check_trial_data=False):
        if check_trial_data and not self.has_trials:
            raise ValueError("{0}: Data has no trial to be manipulated".format(self.class_name))
        if self.has_trials:
            n = self.n_samples / float(self.trial_size)
            if n != int(n):
                raise ValueError("{0}: Invalid data format. Number of samples must be a multiple of the trial length"
                                 .format(self.class_name))
        if self.has_electrodes and len(self.electrodes) != self.n_channels:
            raise ValueError("{0}: Invalid data format".format(self.class_name))
        return True

    def _check_valid_trial_index(self, index):
        if self.has_trials:
            if index < 0:
                raise ValueError("{0}: trial index must be a non-negative number".format(self.class_name))
            elif index >= self.n_trials:
                raise ValueError("{0}: invalid trial index. Data has {1} trials, got index {2}"
                                 .format(self.class_name, self.n_trials, index))
        else:
            if index != 0:
                raise ValueError("{0}: Data has no labeled trials".format(self.class_name))
        return True

    def _get_cache_data(self, key):
        return self._cache.get(key)

    def _set_cache_data(self, key, value):
        self._cache[key] = value
        return self

    def _clone(self, **kwargs):
        d = dict(trial_size=self.trial_size, electrodes=self.electrodes, subject=self.subject,
                 sampling_rate=self.sampling_rate, trial_labels=self.trial_labels, data_reader=self.data_reader)
        d.update(kwargs)
        return EEG(data=d.pop('data', self.data), **d)

    # TODO: accept spherical as a parameter
    def get_elect_coords(self, normalize=False):
        cache_name = 'elect_coord_normal' if normalize else 'elect_coord'
        coord = self._get_cache_data(cache_name)
        if coord is None:
            coord = np.array([[e.get(k, 1) for k in 'xyz'] for e in self.electrodes if isinstance(e, dict)])
        if normalize:
            coord /= np.linalg.norm(coord, axis=1)[:, np.newaxis]
        self._set_cache_data(cache_name, coord)
        return coord

    def get_elect_labels(self):
        elabs = self._get_cache_data('elect_coord')
        if elabs is None:
            elabs = [e.get('label', '') for e in self.electrodes if isinstance(e, dict)]
            self._set_cache_data('elect_coord', elabs)
        return elabs

    def get_trials(self, index_list, new_axis=False):
        if self.n_comps == 0:
            return self._get_trials2d(index_list, new_axis)
        return self._get_trials3d(index_list, new_axis)

    def _get_trials2d(self, index_list, new_axis):
        dta = np.zeros((self.n_channels, self.trial_size, len(index_list)))
        for j, index in enumerate(index_list):
            beg, end = self.get_trial_beg_end(index)
            dta[:, :, j] = self.data[:, beg:end]
        if new_axis:
            return dta
        return dta.reshape((self.n_channels, -1), order='F')

    def _get_trials3d(self, index_list, new_axis):
        dta = np.zeros((self.n_channels, self.trial_size, self.data.shape[2], len(index_list)))
        for j, index in enumerate(index_list):
            beg, end = self.get_trial_beg_end(index)
            dta[:, :, :, j] = self.data[:, beg:end, :]
        if new_axis:
            return dta.transpose((0, 1, 3, 2))
        return dta.reshape((self.n_channels, -1, self.data.shape[2]), order='F')

    def clear_cache(self):
        self._cache = dict()
        return self

    def spatial_transform(self, t, inplace=False, **clone_params):
        if not self.is_potential:
            raise TypeError("{0}: can only transform potential data".format(self.class_name))
        self._check_valid_data_format()
        if inplace:
            self.data = t.transform(self.data)
            return self
        return self._clone(data=t.transform(self.data), **clone_params)

    def get_laplacian(self, lambda_value=1e-2, m=3, truncation=100, inplace=False):
        lap = Laplacian(self.get_elect_coords(normalize=True), m, truncation).eval(lambda_value=lambda_value)
        return self.spatial_transform(lap, inplace=inplace, is_laplacian=True, is_electric_field=False)

    def get_electric_field(self, lambda_value=1e-2, m=3, inplace=False):
        ef = ElectricField(self.get_elect_coords(normalize=True), m).eval(lambda_value=lambda_value)
        return self.spatial_transform(ef, inplace=inplace, is_laplacian=False, is_electric_field=True)

    def get_trial_beg_end(self, index):
        beg = index * self.trial_size
        end = beg + self.trial_size
        return beg, end

    def spectral_decomp(self, min_freq=1, max_freq=50, n_bins=10, inplace=False):
        time_step = 1. / self.sampling_rate
        freqs = np.fft.fftfreq(self.trial_size, d=time_step)
        bins = np.linspace(min_freq, max_freq, n_bins)
        bin_limits = zip(bins[:-1], bins[1:])
        data_new = None
        for n in xrange(self.n_channels):
            y = self.data[n, :]
            ps = np.abs(np.fft.fft(y)) ** 2
            power_bins = np.array([ps[:, (freqs >= f1) & (freqs < f2)].sum(axis=1) for f1, f2 in bin_limits]).T.ravel()[None, :]
            if inplace:
                self.data[n, :len(power_bins)] = power_bins
            else:
                data_new = power_bins if data_new is None else np.r_[data_new, power_bins]
        if inplace:
            self.trial_size = n_bins
            self.data = self.data[:, :(len(self.trial_labels)*self.trial_size)]
            return self
        return self._clone(data=data_new)

    def average_trials(self, group_size, inplace=False):
        if group_size == 1:
            return self
        self._check_valid_data_format(check_trial_data=True)
        groups, self._trial_labels = group_labels(self.trial_labels, group_size)
        data_new = None
        for n, indexes in enumerate(groups):
            x = self.get_trials(indexes, new_axis=True).mean(axis=2)
            if inplace:
                beg, end = self.get_trial_beg_end(n)
                self.data[:, beg:end] = x
                self.group_size = group_size
            else:
                data_new = x if data_new is None else np.c_[data_new, x]
        if inplace:
            self.data = self.data[:, :(len(self.trial_labels) * self.trial_size)]
            return self
        return self._clone(data=data_new)

    def read(self, filename, **kwargs):
        if not file_exists(filename):
            raise IOError("File '{0}' does not exist".format(filename))
        d = self.data_reader(filename, **kwargs)
        self.trial_size = d.get('trial_size', 1)
        self.data = d.get('data', np.array([]))
        self._electrodes = list(d.get('electrodes', []))
        self.subject = d.get('subject', '')
        self.sampling_rate = d.get('sampling_rate', self.DEFAULT_SAMPLING_RATE)
        self._trial_labels = np.asarray(d.get('trial_labels', []))
        self.data_reader = d.get('data_reader', self.data_reader)
        self._is_laplacian = d.get('is_laplacian', False)
        self._is_electric_field = d.get('is_electric_field', False)
        self.group_size = d.get('group_size', 1)
        self.filename = filename
        self._check_valid_data_format()
        self.clear_cache()
        return self
