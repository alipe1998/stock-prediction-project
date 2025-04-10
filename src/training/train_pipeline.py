# src/training/train_pipeline.py

import pandas as pd
import logging
from pathlib import Path
import sys
import joblib
from datetime import datetime
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

# Import the model registry to dynamically select models
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
        # Drop rows with missing values
        
        Xtrain = pooled_data[features]
        ytrain = pooled_data["ret"]
        
        logging.info(f"Pooled data from months: {months_list} with {len(common_tickers)} common tickers.")
        return Xtrain, ytrain
    except Exception as e:
        logging.error(f"Failed pooling training data: {e}")
        raise

def perform_grid_search(model_name, Xtrain, ytrain, param_grid):
    # Dynamically retrieve the model class from the registry
    ModelClass = MODEL_REGISTRY.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Model {model_name} not found in the registry.")
    
    # Instantiate the model with default parameters; grid search will override these
    model_instance = ModelClass()
    
    grid_search = GridSearchCV(estimator=model_instance.model,  # assuming each model wraps the scikit-learn API
                               param_grid=param_grid,
                               cv=5,
                               scoring='r2',
                               n_jobs=-1,
                               verbose=1)
    try:
        grid_search.fit(Xtrain, ytrain)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Save best model with a clear naming scheme
        date_str = datetime.now().strftime("%Y-%m-%d")
        model_filename = f"{model_name}_{date_str}_r2_{best_score:.4f}.pkl"
        joblib.dump(grid_search.best_estimator_, MODEL_DIR / model_filename)
        
        logging.info(f"Best Grid Search Score: {best_score}")
        logging.info(f"Best Params: {best_params}")
        logging.info(f"Model saved as: {MODEL_DIR / model_filename}")

        return best_params, best_score, grid_search.best_estimator_
    
    except Exception as e:
        logging.error(f"Grid search failed: {e}")
        raise
