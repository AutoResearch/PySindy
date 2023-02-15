import pysindy

from inspect import signature
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SINDyRegressor(BaseEstimator, RegressorMixin):
    """
    PySINDy Wrapper
    PySINDy is a pypi package
    PySINDy repository can be found at: https://github.com/dynamicslab/pysindy

    Attributes:

    """

    def __init__(
        self,
    ):
        """
        Arguments:

        """
        self.X_ = None
        self.y_ = None
        self.timesteps = None
        self.variables = list()
        self.model_ = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            **kwargs: dict):
        """
        Runs the optimization for a given set of `X`s and `y`s.

        Arguments:
            X: independent variables in an n-dimensional array
            y: dependent variables in an n-dimensional array

        Returns:
            self (PySINDy): the fitted estimator
        """
        # firstly, store the column names of X since checking will
        # cast the type of X to np.ndarray
        if hasattr(X, "columns"):
            self.variables = list(X.columns)
        else:
            # create variables X_1 to X_n where n is the number of columns in X
            self.variables = ["X%d" % i for i in range(X.shape[1])]

        X, y = check_X_y(X, y)

        args = {
            arg: kwargs[arg] for arg in kwargs.keys() if kwargs[arg] in signature(pysindy.SINDy)
        }
        XandY = np.stack([X, y], axis=-1)
        XandY = XandY / np.linalg.norm(XandY)
        self.timesteps = np.ndarray(shape=(XandY.shape[0],))
        # This is where PySINDy backend starts
        self.model_ = pysindy.SINDy(**args)
        self.model_.fit(XandY, t=self.timesteps)
        self.X_, self.y_ = X, y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the fitted model to a set of independent variables `X`,
        to give predictions for the dependent variable `y`.

        Arguments:
            X: independent variables in an n-dimensional array

        Returns:
            y: predicted dependent variable values
        """
        # this validation step will cast X into np.ndarray format
        X = check_array(X)



        check_is_fitted(self, attributes=["model_"])

        assert self.model_ is not None

        return self.model_.simulate(X, t=self.timesteps)
