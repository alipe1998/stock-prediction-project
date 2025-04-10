# src/backtest/backtest.py
from pathlib import Path
import sys
import pandas as pd
import logging
from datetime import datetime
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)

# Define the path for saving models
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

MODEL_DIR = ROOT_DIR / "data" / "models"

from data.preprocess import preprocess_data
from training.train_pipeline import pool_training_data, perform_grid_search

logging.basicConfig(level=logging.INFO)

def run_backtest(df, model_name, features, months_back, start_month, end_month, param_grid, preprocessing_pipeline):
    """
    Runs a backtest using a rolling window approach.
    
    Parameters:
      df: DataFrame containing monthly stock data with 'date', 'ticker', and 'ret' columns.
      model_name: String name of the model ('mlp' or 'xgboost') to be used.
      features: List of feature names to use.
      months_back: Number of months of historical data to pool for training.
      start_month: Start date of backtesting period in 'YYYY-MM' format.
      end_month: End date of backtesting period in 'YYYY-MM' format.
      param_grid: Dictionary of hyperparameters for grid search.
      
    Returns:
      results_df: DataFrame with backtesting results for each period.
    """
    results = []
    
    # Create a date range for the training period endpoints
    training_end_dates = pd.date_range(start=start_month, end=end_month, freq='MS').strftime('%Y-%m')
    
    for current_month in training_end_dates:
        # Pool training data from the past `months_back` months up to current_month.
        try:
            X_train, y_train = pool_training_data(df, current_month, months_back, features)
            X_train = preprocess_data(X_train, features, preprocessing_pipeline)
            # isnpect industry columns
            industry_cols = [col for col in X_train.columns if col.startswith("ind_")]
            y_train = y_train.loc[X_train.index]
        except Exception as e:
            logging.warning(f"Skipping month {current_month}: error pooling training data: {e}")
            continue

        # Train the model using grid search
        try:
            best_params, train_r2, best_model = perform_grid_search(model_name, X_train, y_train, param_grid)
        except Exception as e:
            logging.warning(f"Skipping month {current_month}: error during grid search: {e}")
            continue

        # Define test month as the month following current_month
        test_date = (datetime.strptime(current_month, '%Y-%m') + pd.DateOffset(months=1)).strftime('%Y-%m')
        test_data = df[df['date'] == test_date]
        if test_data.empty:
            logging.warning(f"No test data available for {test_date}, skipping.")
            continue

        X_test = test_data[features]
        y_test = test_data["ret"]

        X_test = preprocess_data(X_test, features, preprocessing_pipeline)
        y_test = y_test.loc[X_test.index]
        # Make predictions and evaluate with R² score
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        logging.info(f"Backtest for test month {test_date}: Train R2 = {train_r2:.4f}, Test R2 = {test_r2:.4f}")
        
        # Build lists for detailed prediction records
        tickers = []
        actual_returns = []
        predicted_returns = []
        predictions = []
        test_data_filtered = test_data.loc[X_test.index]
        for idx, row in test_data_filtered.iterrows():
            tickers.append(row["ticker"])
            actual_returns.append(row["ret"])
            # Get corresponding predicted return (assumes same order as X_test.index)
            pred_value = y_pred[list(X_test.index).index(idx)]
            predicted_returns.append(pred_value)
            predictions.append({
                "ticker": row["ticker"],
                "actual_ret": row["ret"],
                "predicted_ret": pred_value
            })

        results.append({
            "train_end": current_month,
            "test_month": test_date,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "best_params": best_params,
            "predictions": predictions,
            "tickers": tickers,
            "actual_returns": actual_returns,
            "predicted_returns": predicted_returns
        })

    results_df = pd.DataFrame(results)
    return results_df
