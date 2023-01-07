import numpy as np
from sklearn.cross_decomposition import CCA

from bmi.estimators.function_wrapper import _EmptyParams
from bmi.interface import BaseModel, IMutualInformationPointEstimator


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return np.corrcoef(x, y)[0, 1]


def mi_gauss(correlation: float) -> float:
    return -0.5 * np.log(1 - correlation**2)


class CCAMutualInformationEstimator(IMutualInformationPointEstimator):
    def __init__(self, scale: bool = True) -> None:
        self._scale = scale

    def estimate(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        n_components = min(x.shape[-1], y.shape[-1])
        x_new, y_new = CCA(n_components=n_components, scale=self._scale).fit_transform(x, y)
        dims = x_new.shape[-1]
        assert y_new.shape[-1] == dims

        return sum([mi_gauss(corr(x_new[:, i], y_new[:, i])) for i in range(dims)])

    def parameters(self) -> BaseModel:
        return _EmptyParams()
