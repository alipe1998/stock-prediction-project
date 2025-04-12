# src/backtest/backtest.py
from pathlib import Path
import sys
import pandas as pd
import logging
from datetime import datetime
from sklearn.metrics import r2_score
import time


logging.basicConfig(level=logging.INFO)

# Define the path for saving models
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

MODEL_DIR = ROOT_DIR / "data" / "models"

from data.preprocess import preprocess_data
from training.train_pipeline import pool_training_data, perform_grid_search

logging.basicConfig(level=logging.INFO)

def run_backtest(df, model_name, features, months_back, start_month, end_month, param_grid, preprocessing_pipeline, tune_hyperparams=False):
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
    total_months = len(training_end_dates)
    logging.info(f"Starting backtest from {start_month} to {end_month} across {total_months} months.")

    for i, current_month in enumerate(training_end_dates, start=1):
        month_start_time = time.perf_counter()
        logging.info(f"\n[{i}/{total_months}] >>> Processing month: {current_month}")

        try:
            # Pool and preprocess training data
            X_train, y_train = pool_training_data(df, current_month, months_back, features)
            X_train = preprocess_data(X_train, features, preprocessing_pipeline)
            y_train = y_train.loc[X_train.index]
        except Exception as e:
            logging.warning(f"Skipping month {current_month}: error pooling/preprocessing training data: {e}")
            continue

        try:
            # Train model and get best hyperparams and training performance
            best_params, train_r2, best_model = perform_grid_search(model_name, X_train, y_train, param_grid, tune_hyperparams=tune_hyperparams)
        except Exception as e:
            logging.warning(f"Skipping month {current_month}: error during model training/grid search: {e}")
            continue

        # Determine test month
        test_date = (datetime.strptime(current_month, '%Y-%m') + pd.DateOffset(months=1)).strftime('%Y-%m')
        test_data = df[df['date'] == test_date]
        if test_data.empty:
            logging.warning(f"No test data available for {test_date}, skipping.")
            continue

        try:
            # Prepare test data
            X_test = test_data[features]
            y_test = test_data["ret"]
            X_test = preprocess_data(X_test, features, preprocessing_pipeline)
            y_test = y_test.loc[X_test.index]

            # Predict and evaluate
            y_pred = best_model.predict(X_test).flatten()
            test_r2 = r2_score(y_test, y_pred)
            logging.info(f"✅ Backtest {test_date}: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")

            # Save detailed results
            predictions = []
            for idx, row in test_data.loc[X_test.index].iterrows():
                pred_value = y_pred[list(X_test.index).index(idx)]
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
                "predictions": predictions
            })

        except Exception as e:
            logging.warning(f"Skipping month {current_month}: error during test evaluation: {e}")
            continue

        month_end_time = time.perf_counter()
        elapsed = month_end_time - month_start_time
        logging.info(f"⏱ Finished month {current_month} in {elapsed:.2f} seconds")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logging.warning("Backtest completed, but no valid results were collected.")
    else:
        logging.info(f"\n✅ Backtest complete. Processed {len(results_df)} months with results.")
    
    return results_df
