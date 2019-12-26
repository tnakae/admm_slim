import numpy as np
from scipy import sparse

from .base import BaseSlimModel


class DenseSlim(BaseSlimModel):
    def __init__(self, lambda_2=500):
        self.lambda_2 = lambda_2

    def fit(self, X):
        XtX = X.T.dot(X)
        if sparse.issparse(XtX):
            XtX = XtX.todense().A

        identity_mat = np.identity(XtX.shape[0])
        diags = identity_mat * self.lambda_2
        P = np.linalg.inv(XtX + diags)
        self.coef = identity_mat - P.dot(np.diag(1. / np.diag(P)))
