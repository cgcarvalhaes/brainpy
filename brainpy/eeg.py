import hashlib
import json

import numpy as np
from eeg_reshape import eeg_reshape
from etc import regroup_labels
from laplacian import Laplacian
from utils import file_exists, json_default

from .electric_field import ElectricField


class EEG(object):
    DEFAULT_SAMPLING_RATE = 1e3
    DERIVATIONS = ['potential', 'laplacian', 'electric_field']

    def __init__(self, data=None, trial_size=None, electrodes=None, subject=None, sampling_rate=DEFAULT_SAMPLING_RATE,
                 trial_labels=None, data_reader=None, is_laplacian=False, is_electric_field=False, filename=None,
                 lambda_value=1e-2, interpol_order=3, truncation=100):
        self.trial_size = trial_size
        self.data = data
        if self.data is None:
            self.data = np.empty((1, 1, 1))
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
        self.der_code = 0
        self.filename = filename
        self.group_size = 1
        self.lambda_value = lambda_value
        self.interpol_order = interpol_order
        self.truncation = truncation

    @property
    def doc(self):
        return {
            "trial_size": json_default(self.trial_size),
            "data_shape": json_default(self.data.shape),
            "electrodes": self.electrodes,
            "subject": json_default(self.subject),
            "sampling_rate": self.sampling_rate,
            "trial_labels": self.trial_labels,
            "filename": self.filename,
            "group_size": self.group_size,
            "n_channels": self.n_channels,
            "n_trials": self.n_trials,
            "n_comps": self.n_comps,
            "derivation": self.derivation,
            "lambda_value": self.lambda_value,
            "interpol_order": self.interpol_order,
            "truncation": self.truncation
        }

    def __str__(self):
        return '%s: ' % self.derivation + object.__str__(self)

    @property
    def identifier(self):
        word = json.dumps(self.doc, default=json_default)
        return hashlib.md5(word).hexdigest()

    @property
    def derivation(self):
        return self.DERIVATIONS[self.der_code]

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
        return self.data.shape[1]

    @property
    def n_trials(self):
        if self.trial_size:
            return self.n_samples // self.trial_size
        return 1

    @property
    def n_comps(self):
        return self.data.shape[2]

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
        return self.der_code == 0

    @property
    def is_laplacian(self):
        return self.der_code == 1

    @property
    def is_electric_field(self):
        return self.der_code == 2

    def _check_valid_data_format(self, check_trial_data=False):
        if check_trial_data and not self.has_trials:
            raise ValueError("{0}: Data has no trials to be manipulated".format(self.class_name))
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
        elif index != 0:
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

    # TODO: accept spherical coordinates as a parameter
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
        if new_axis:
            dta = np.zeros((self.n_channels, self.trial_size, self.n_comps, len(index_list)))
            for j, index in enumerate(index_list):
                dta[:, :, :, j] = self.get_single_trial(index)
        else:
            dta = np.zeros((self.n_channels, self.trial_size * len(index_list), self.n_comps))
            for j, index in enumerate(index_list):
                k = j * self.trial_size
                dta[:,  k:(k + self.trial_size), :] = self.get_single_trial(index)
        return dta

    def get_single_trial(self, index):
        beg, end = self.get_trial_beg_end(index)
        return self.data[:, beg:end, :]

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

    def get_laplacian(self, inplace=False):
        lap = Laplacian(self.get_elect_coords(normalize=True), self.interpol_order, truncation=self.truncation)\
            .eval(lambda_value=self.lambda_value)
        return self.spatial_transform(lap, inplace=inplace, der_code=1)

    def get_electric_field(self, inplace=False):
        ef = ElectricField(self.get_elect_coords(normalize=True), self.interpol_order, truncation=self.truncation)\
            .eval(lambda_value=self.lambda_value)
        return self.spatial_transform(ef, inplace=inplace, der_code=2)

    def get_trial_beg_end(self, index):
        beg = index * self.trial_size
        end = beg + self.trial_size
        return beg, end

    def get_channels(self, channel_list):
        return self.data[channel_list, :]

    def average_trials(self, group_size, inplace=False):
        if group_size == 1:
            return self
        self._check_valid_data_format(check_trial_data=True)
        groups, self._trial_labels = regroup_labels(self.trial_labels, group_size)
        if inplace:
            for n, indexes in enumerate(groups):
                x = self.get_trials(indexes, new_axis=True).mean(axis=3)
                beg, end = self.get_trial_beg_end(n)
                self.data[:, beg:end, :] = x
                self.group_size = group_size
            self.data = self.data[:, :(len(self.trial_labels) * self.trial_size)]
            return self
        else:
            data_new = np.zeros((self.n_channels, self.trial_size * len(self._trial_labels), self.n_comps))
            for n, indexes in enumerate(groups):
                x = self.get_trials(indexes, new_axis=True).mean(axis=3)
                beg, end = self.get_trial_beg_end(n)
                data_new[:, beg:end, :] = x
            return self._clone(data=data_new)

    def to_clf_format(self, channels=None):
        if channels is None:
            channels = range(self.n_channels)
        if isinstance(channels, (int, np.int)):
            channels = [channels]
        return eeg_reshape(self.data[channels, :, :], self.trial_size)

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
        self.der_code = d.get('der_code', 0)
        self.group_size = d.get('group_size', 1)
        self.filename = filename
        self._check_valid_data_format()
        self.clear_cache()
        return self

    # TODO: It will be turned into a feature extraction object for classification at some point
    def spectral_decomp(self, min_freq=1, max_freq=50, n_bins=10, inplace=False):
        time_step = 1. / self.sampling_rate
        freqs = np.fft.fftfreq(self.trial_size, d=time_step)
        bins = np.linspace(min_freq, max_freq, n_bins)
        bin_limits = zip(bins[:-1], bins[1:])
        data_new = None
        for n in xrange(self.n_channels):
            y = self.data[n, :]
            ps = np.abs(np.fft.fft(y)) ** 2
            power_bins = np.array([ps[:, (freqs >= f1) & (freqs < f2)].sum(axis=1) for f1, f2 in bin_limits]).T.ravel()[
                         None, :]
            if inplace:
                self.data[n, :len(power_bins)] = power_bins
            else:
                data_new = power_bins if data_new is None else np.r_[data_new, power_bins]
        if inplace:
            self.trial_size = n_bins
            self.data = self.data[:, :(len(self.trial_labels) * self.trial_size)]
            return self
        return self._clone(data=data_new)

