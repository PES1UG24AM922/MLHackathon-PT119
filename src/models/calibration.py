"""Persist a fitted scorer and a trip-isolated sigmoid probability calibration."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def log_odds(probabilities):
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return (np.log(p) - np.log1p(-p)).reshape(-1, 1)


def fit_sigmoid(probabilities, labels):
    return LogisticRegression(random_state=42).fit(log_odds(probabilities), labels)


class SigmoidScorer:
    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, features):
        raw = self.estimator.predict_proba(features)[:, 1]
        return self.calibrator.predict_proba(log_odds(raw))
