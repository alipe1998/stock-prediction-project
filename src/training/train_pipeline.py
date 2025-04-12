# src/training/train_pipeline.py

import pandas as pd
import logging
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from config.model_registry import MODEL_REGISTRY

# Define the path for saving models
MODEL_DIR = ROOT_DIR / "data" / "models"

def pool_training_data(df, current_month, months_back, features):
    try:
        current_date = pd.to_datetime(current_month, format='%Y-%m')
        months_list = [(current_date - pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(months_back)]
        
        ticker_sets = [set(df[df.date == month]['ticker']) for month in months_list]
        common_tickers = set.intersection(*ticker_sets)
        
        pooled_data = df[df.date.isin(months_list) & df.ticker.isin(common_tickers)]
        Xtrain = pooled_data[features]
        ytrain = pooled_data["ret"]
        
        logging.info(f"Pooled data from months: {months_list} with {len(common_tickers)} common tickers.")
        return Xtrain, ytrain
    except Exception as e:
        logging.error(f"Failed pooling training data: {e}")
        raise

def perform_grid_search(model_name, Xtrain, ytrain, param_grid):
    """
    Manual grid search over the hyperparameters in `param_grid`. For each combination,
    the training data is split into train and validation parts. The model is trained on the 
    training split and evaluated on the validation split using R² score.
    
    After finding the best hyperparameters, the best model is retrained on the full training data.
    
    Returns:
      best_params: Dictionary of the best hyperparameters.
      train_r2: R² score on the full training set.
      best_model: The trained model instance.
    """
    # Retrieve the model class (should be MLPKerasModel)
    ModelClass = MODEL_REGISTRY.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Model {model_name} not found in the registry.")
    
    param_names = list(param_grid.keys())
    param_values_list = [param_grid[p] for p in param_names]
    
    best_score = -np.inf
    best_params = None
    best_model = None
    
    # Split training data into train and validation sets (80/20 split)
    X_train_split, X_val, y_train_split, y_val = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=42)
    
    for param_combination in product(*param_values_list):
        current_params = dict(zip(param_names, param_combination))
        input_shape = (Xtrain.shape[1],)
        model_instance = ModelClass(input_shape=input_shape, **current_params)
        # Train on the training split
        model_instance.train(X_train_split, y_train_split, verbose=0)
        # Predict on the validation split
        y_pred_val = model_instance.predict(X_val).flatten()
        current_score = r2_score(y_val, y_pred_val)
        logging.info(f"Tested params {current_params} => val R2: {current_score:.4f}")
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            best_model = model_instance
    
    # Retrain the best model on the full training data using the best hyperparameters.
    input_shape = (Xtrain.shape[1],)
    best_model = ModelClass(input_shape=input_shape, **best_params)
    best_model.train(Xtrain, ytrain, verbose=0)
    
    logging.info(f"Best hyperparameters: {best_params}, validation R2: {best_score:.4f}")
    
    # Save best model using Keras native save (file will be in .h5 format)
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_filename = f"{model_name}_{date_str}_r2_{best_score:.4f}.h5"
    save_path = MODEL_DIR / model_filename
    best_model.save(save_path)
    logging.info(f"Model saved as: {save_path}")
    
    # Evaluate training performance on full training set
    y_pred_train = best_model.predict(Xtrain).flatten()
    train_r2 = r2_score(ytrain, y_pred_train)
    
    return best_params, train_r2, best_model
