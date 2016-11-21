import numpy as np


def get_eeg_shape(data, trial_size):
    n_channels = data.shape[0]
    n_trials = data.shape[1] // trial_size
    n_comps = data.shape[2]
    n_samples = n_trials * trial_size
    assert n_trials * trial_size == data.shape[1]
    return n_channels, n_trials, n_comps, n_samples


def eeg_reshape(data, trial_size):
    n_chans, n_trials, n_comps, n_samples = get_eeg_shape(data, trial_size)
    data = np.array([data[:, i:(i + trial_size), :].ravel() for i in range(0, n_samples, trial_size)])
    return data
