import numpy as np

from eeg import EEG
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class EEGClassify(EEG):
    def __init__(self, clf, test_proportion, random_state=42, **kwargs):
        super(EEGClassify, self).__init__(**kwargs)
        self.random_state = random_state
        self.test_proportion = test_proportion
        self.clf = clf
        self.test = np.array([])
        self.predictions = np.array([])
        self.scores = np.array([])

    def classify(self, verbose=True):
        for ch in range(self.n_channels):
            x_train, x_test, y_train, y_test = train_test_split(self.data[ch, :].reshape((-1, self.trial_size)),
                                                                   self.trial_labels.astype(np.int32),
                                                                   test_size=self.test_proportion,
                                                                   random_state=self.random_state)
            if ch == 0:
                self.test = np.zeros((self.n_channels, len(y_test)))
                self.predictions = np.zeros_like(self.test)
                self.scores = np.zeros(self.n_channels)
            self.clf.fit(x_train, y_train)
            self.test = y_test
            self.predictions[ch, :] = self.clf.predict(x_test)
            self.scores[ch] = accuracy_score(y_test, self.predictions[ch])
            if verbose:
                print "Channel %s: %.0f%%" % (ch, 100*self.scores[ch])
        return self
