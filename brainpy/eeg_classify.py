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

    def classify(self):
        result = {}
        for ch in range(self.n_channels):
            x_train, x_test, y_train, y_test = train_test_split(self.data[ch, :].reshape((-1, self.trial_size)),
                                                                self.trial_labels.astype(np.int32),
                                                                test_size=self.test_proportion,
                                                                random_state=self.random_state)
            self.clf.fit(x_train, y_train)
            y_pred = self.clf.predict(x_test)
            score = accuracy_score(y_test, y_pred)
            result[ch] = score
            print "Channel {0}:\t{1}%%".format(ch, 100*score)
        return result
