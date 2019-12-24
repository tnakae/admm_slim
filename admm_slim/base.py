import abc

class BaseSlimModel(abc.ABC):
    @abc.abstractclassmethod
    def fit(self, X):
        pass

    @abc.abstractclassmethod
    def predict(self, X):
        pass

    @abc.abstractclassmethod
    def recommend(self, X):
        pass
