# src/training/train_pipeline.py

import pandas as pd
import logging
from pathlib import Path
import sys
import time
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

def perform_grid_search(model_name, Xtrain, ytrain, param_grid, tune_hyperparams=True):
    """
    Manual grid search over the hyperparameters in `param_grid` if tuning is enabled.
    For each combination, the training data is split into train and validation parts.
    The model is trained on the training split and evaluated on the validation split using R².
    
    If tuning is disabled, the function uses the first candidate for each parameter.
    
    After selecting the parameters (either via grid search or defaults), the best model
    is retrained on the full training data.
    
    Returns:
      best_params: Dictionary of chosen hyperparameters.
      train_r2: R² score on the full training set.
      best_model: The trained model instance.
    """
    # Retrieve the model class (should be MLPKerasModel)
    ModelClass = MODEL_REGISTRY.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Model {model_name} not found in the registry.")
    
    param_names = list(param_grid.keys())
    
    # If hyperparameter tuning is disabled, use the first candidate from each parameter list.
    if not tune_hyperparams:
        default_params = {p: param_grid[p][0] for p in param_names}
        logging.info(f"Hyperparameter tuning disabled. Using default hyperparameters: {default_params}")
        input_shape = (Xtrain.shape[1],)
        start_time = time.perf_counter()
        model_instance = ModelClass(input_shape=input_shape, **default_params)
        model_instance.train(Xtrain, ytrain, verbose=0)
        end_time = time.perf_counter()
        logging.info(f"Training with default hyperparameters completed in {end_time - start_time:.2f} seconds")
        y_pred_train = model_instance.predict(Xtrain).flatten()
        train_r2 = r2_score(ytrain, y_pred_train)
        return default_params, train_r2, model_instance
    
    # Otherwise, proceed with grid search.
    param_values_list = [param_grid[p] for p in param_names]
    best_score = -np.inf
    best_params = None
    best_model = None
    
    # Split training data into train and validation sets (80/20 split)
    X_train_split, X_val, y_train_split, y_val = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=42)
    
    logging.info(f"Beginning grid search for model: {model_name}")
    total_combinations = len(list(product(*param_values_list)))
    
    for i, param_combination in enumerate(product(*param_values_list), start=1):
        current_params = dict(zip(param_names, param_combination))
        input_shape = (Xtrain.shape[1],)
        logging.info(f"[{i}/{total_combinations}] Trying params: {current_params}")
        
        try:
            model_instance = ModelClass(input_shape=input_shape, **current_params)
            start_time = time.perf_counter()
            model_instance.train(X_train_split, y_train_split, verbose=0)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            logging.info(f"→ Trained in {elapsed:.2f} seconds")
            
            y_pred_val = model_instance.predict(X_val).flatten()
            current_score = r2_score(y_val, y_pred_val)
            logging.info(f"→ R² on validation set: {current_score:.4f}")
        except Exception as e:
            logging.warning(f"Model failed with params {current_params}: {e}")
            continue
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            best_model = model_instance
            logging.info(f"↑ New best score: {best_score:.4f} with params {best_params}")
    
    logging.info(f"Best hyperparameters: {best_params}, validation R²: {best_score:.4f}")
    
    # Retrain the best model on the full training data using the best hyperparameters.
    logging.info("Retraining best model on full training data...")
    start_time = time.perf_counter()
    input_shape = (Xtrain.shape[1],)
    best_model = ModelClass(input_shape=input_shape, **best_params)
    best_model.train(Xtrain, ytrain, verbose=0)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    logging.info(f"Retraining completed in {elapsed:.2f} seconds")
    
    # Evaluate training performance on the full training set.
    y_pred_train = best_model.predict(Xtrain).flatten()
    train_r2 = r2_score(ytrain, y_pred_train)
    
    return best_params, train_r2, best_model
