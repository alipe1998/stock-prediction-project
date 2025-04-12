# inference.py

import os
import sys
import logging
import joblib
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src"))

load_dotenv(ROOT_DIR / ".env")

from data.fetch_data import get_db_engine, fetch_stock_data
from data.preprocess import preprocess_data
# Define paths

PREDICTION_OUTPUT_PATH = ROOT_DIR / "data" / "live" / f"predictions_{datetime.now().date()}.csv"

# Load database credentials
server = os.getenv("DB_SERVER")
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")

# Ensure environment variables are loaded
assert all([server, username, password, database]), "Missing DB credentials in .env"

def load_trained_model(model_path):
    try:
        model = joblib.load(model_path)
        logging.info("Trained model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def get_latest_data(engine, features, current_month):
    query = f"""
        SELECT ticker, date, {', '.join(features)}, ret
        FROM data
        WHERE date = '{current_month}'
    """
    df = fetch_stock_data(engine, query)
    if df.empty:
        raise ValueError(f"No data found for date {current_month}")
    return df

def generate_predictions(model, df, features):
    X_live = df[features]
    predictions = model.predict(X_live)
    df_predictions = df[['ticker', 'date']].copy()
    df_predictions["predicted_ret"] = predictions
    return df_predictions

def save_predictions(df, output_path=PREDICTION_OUTPUT_PATH):
    df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

def run_inference(model_path, features, current_month):
    engine = get_db_engine(server, username, password, database)

    # Load model
    model = load_trained_model(model_path=model_path)

    # Fetch latest data
    df_live = get_latest_data(engine, features, current_month)

    # Preprocess data (consistent with training)
    df_live_clean = preprocess_data(df_live, features)

    # Generate predictions
    predictions_df = generate_predictions(model, df_live_clean, features)

    # Save predictions
    save_predictions(predictions_df)

    return predictions_df
