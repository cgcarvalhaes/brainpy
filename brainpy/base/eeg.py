import numpy as np

from spatial.electric_field import ElectricField


# TODO: check if data reader is a valid function
# TODO: garbage collector
from spatial.laplacian import Laplacian
from utils.utils import group_labels


class EEG(object):
    def __init__(self, data=None, trial_size=None, electrodes=None, subject=None, sampling_rate=1e3, trial_labels=None,
                 data_reader=None, is_laplacian=False, is_electric_field=False):
        self.trial_size = trial_size
        self.data = data
        if self.data is None:
            self.data = np.array([])
        self.electrodes = electrodes
        if self.electrodes is None:
            self.electrodes = []
        self.subject = subject
        self.sampling_rate = sampling_rate
        self.trial_labels = trial_labels
        if self.trial_labels is None:
            self.trial_labels = np.array([])
        self.data_reader = data_reader
        self._cache = dict()
        self._is_laplacian = is_laplacian
        self._is_electric_field = is_electric_field

    # TODO: allow coordinate transformations (e.g., spherical)
    @property
    def get_elect_coords(self):
        ecoords = self._get_cache_data('elect_coord')
        if ecoords is None:
            ecoords = np.array([[e.get(k, np.nan) for k in 'xyz'] for e in self.electrodes if isinstance(e, dict)])
            self._set_cache_data('elect_coord', ecoords)
        return ecoords

    @property
    def get_elect_labels(self):
        elabs = self._get_cache_data('elect_coord')
        if elabs is None:
            elabs = [e.get('label', '') for e in self.electrodes if isinstance(e, dict)]
            self._set_cache_data('elect_coord', elabs)
        return elabs

    @property
    def n_channels(self):
        return len(self.data)

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
    def this(self):
        return self.__class__.__name__

    @property
    def has_trials(self):
        return self.n_trials > 1

    @property
    def has_electrodes(self):
        return len(self.electrodes) > 0

    def _check_valid_data_format(self, check_trial_data=False):
        if check_trial_data and not self.has_trials:
            raise ValueError("{0}: Data has no trial to be manipulated".format(self.this))
        if self.has_trials:
            n = self.n_samples / float(self.trial_size)
            if n != int(n):
                raise ValueError("{0}: Invalid data format. Number of samples must be a multiple of the trial length"
                                 .format(self.this))
        if self.has_electrodes and len(self.electrodes) != self.n_channels:
            raise ValueError("{0}: Invalid data format".format(self.this))
        return True

    def _check_valid_trial_index(self, index):
        if self.has_trials:
            if index < 0:
                raise ValueError("{0}: trial index must be a non-negative number".format(self.this))
            elif index >= self.n_trials:
                raise ValueError("{0}: invalid trial index. Data has {1} trials, got index {2}"
                                 .format(self.this, self.n_trials, index))
        else:
            if index != 0:
                raise ValueError("{0}: Data has no labeled trials".format(self.this))
        return True

    def _check_potential(self):
        if self._is_laplacian or self._is_electric_field:
            raise TypeError("{0}: transformation can only be applied to the electric potential".format(self.this))

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

    def get_trials(self, index_list, new_axis=False):
        dta = np.zeros((self.n_channels, self.trial_size, len(index_list)))
        for j, index in enumerate(index_list):
            beg, end = self.get_trial_beg_end(index)
            dta[:, :, j] = self.data[beg:end]
        if new_axis:
            return dta
        return dta.reshape((self.n_channels, -1), order='F')

    def clear_cache(self):
        self._cache = dict()
        return self

    def get_laplacian(self, lambda_value=1e-2, m=3, truncation=100, inplace=False):
        self._check_potential()
        self._check_valid_data_format()
        self.normalize_elect_coords()
        lap = Laplacian(self.get_elect_coords(), m, truncation).eval(lambda_value=lambda_value)
        if inplace:
            self.data = lap.transform(self.data)
            return self
        return self._clone(data=lap.transform(self.data), is_laplacian=True, is_electric_field=False)

    def get_electric_field(self, lambda_value=1e-2, m=3, inplace=False):
        self._check_potential()
        self._check_valid_data_format()
        self.normalize_elect_coords()
        ef = ElectricField(self.get_elect_coords(), m).eval(lambda_value=lambda_value)
        if not self.has_trials:
            if inplace:
                self.data = ef.transform(self.data)
                return self
            else:
                return self._clone(data=ef.transform(self.data), is_laplacian=False, is_electric_field=True)

        data_new = None
        for k in xrange(self.n_trials):
            beg, end = self.get_trial_beg_end(k)
            trial = np.c_[ef[:self.n_channels, beg:end],
                          ef[self.n_channels:2 * self.n_channels, beg:end],
                          ef[2 * self.n_channels:, beg:end]]
            data_new = trial if data_new is None else np.c_[data_new, trial]
        if inplace:
            self.data = data_new
            self.trial_size *= 3
            return self
        return self._clone(data=data_new, is_laplacian=False, is_electric_field=True)

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
        self._check_valid_data_format(check_trial_data=True)
        groups, self.trial_labels = group_labels(self.trial_labels, group_size)
        if inplace:
            self.data = self.data[:, :(len(self.trial_labels)*self.trial_size)]
        data_new = None
        for n, indexes in enumerate(groups):
            x = self.get_trials(indexes, new_axis=True).mean(axis=2)
            if inplace:
                beg, end = self.get_trial_beg_end(n)
                self.data[:, beg:end] = x
            else:
                data_new = x if data_new is None else np.c_[data_new, x]
        if inplace:
            return self
        return self._clone(data=data_new)

    def normalize_elect_coords(self):
        for e in self.get_elect_coords:
            r = np.linalg.norm([e['x'], e['y'], e['z']])
            e.update({'x': e['x'] / r, 'y': e['y'] /r, 'z': e['z'] / r})
        # call get_elect_coords to update the cache
        self.get_elect_coords()
        return self

    # TODO: check if filename is valid
    # TODO: check keys before updating
    def read(self, filename):
        self.__dict__.update(self.data_reader(filename))
        return self
