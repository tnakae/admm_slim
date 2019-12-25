import numpy as np
from scipy import sparse

from .base import BaseSlimModel


class AdmmSlim(BaseSlimModel):
    def __init__(self, lambda_1=1, lambda_2=500, rho=10000,
                 positive=True, n_iter=50, eps_abs=1e-3, eps_rel=1e-4):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.rho = rho
        self.positive = positive
        self.n_iter = n_iter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

    def _thresholding(self, B, gamma):
        x = B + gamma / self.rho
        threshold = self.lambda_1 / self.rho
        return x / np.abs(x) * np.maximum(x - threshold, 0)

    def fit(self, X):
        XtX = X.T.dot(X)
        if sparse.issparse(XtX):
            XtX = XtX.todense().A

        identity_mat = np.identity(XtX.shape[0])
        diags = identity_mat * (self.lambda_2 + self.rho)
        P = np.linalg.inv(XtX + diags)
        B_aux = P.dot(XtX)

        gamma = np.zeros_like(XtX)
        C = np.zeros_like(XtX)

        for iter in range(n_iter):
            B_tilde = B_aux + P.dot(rho * C - gamma)
            gamma = np.diag(B_tilde) / np.diag(P)
            B = B_tilde - P * gamma

            self.B = identity_mat - P.dot(np.diag(1. / np.diag(P)))

    def predict_proba(self, X):
        pass

    def recommend(self, X, top=20):
        pass
