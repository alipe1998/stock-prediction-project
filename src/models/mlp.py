# src/models/mlp.py
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
from .base_model import BaseModel

class MLPModel(BaseModel):
    def __init__(self, **params):
        self.model = MLPRegressor(**params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        dump(self.model, path)

    def load(self, path):
        self.model = load(path)
