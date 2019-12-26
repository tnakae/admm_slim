import numpy as np


class BaseSlimModel(object):
    def fit(self, X):
        self.coef = np.identity(X.shape[1])

    def predict(self, X):
        return X.dot(self.coef)

    def recommend(self, X, top=20):
        scores = self.predict(X)
        top_items = np.argsort(scores, axis=1)[:, -top:]
        return top_items
