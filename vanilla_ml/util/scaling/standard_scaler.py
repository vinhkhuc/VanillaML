from vanilla_ml.util.scaling.abstract_scaler import AbstractScaler


class StandardScaler(AbstractScaler):

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std[None, :]
