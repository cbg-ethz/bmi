"""Base classes for the estimators."""


class EstimatorNotFittedException(Exception):
    """Exception raised when the estimator needs to be fitted first (to access some method)."""

    pass
