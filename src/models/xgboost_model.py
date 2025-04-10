# src/models/xgboost_model.py

import xgboost as xgb
from joblib import dump, load
from .base_model import BaseModel

class XGBModel(BaseModel):
    def __init__(self, **params):
        # Initialize an XGBoost Regressor with the provided parameters
        self.model = xgb.XGBRegressor(**params)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        dump(self.model, path)
    
    def load(self, path):
        self.model = load(path)
