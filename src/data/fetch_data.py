import pandas as pd
import logging
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)

def get_db_engine(server, username, password, database):
    try:
        connection_string = f"mssql+pymssql://{username}:{password}@{server}/{database}"
        engine = create_engine(connection_string)
        logging.info("Database connection established successfully.")
        return engine
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

def fetch_stock_data(engine, query):
    try:
        df = pd.read_sql(query, engine)
        logging.info(f"Fetched {len(df)} rows from database.")
        return df
    except Exception as e:
        logging.error(f"Failed to execute query: {e}")
        raise

def fetch_stock_data_with_features(engine, features, date_config, table_name="data"):
    """
    Fetches stock data using a dynamically built SQL query based on features provided,
    filtering the data using the overall date range specified in date_config.

    Parameters:
      engine: SQLAlchemy engine instance.
      features: List of feature names (strings) to fetch along with ticker, date, and ret.
      date_config: Dictionary containing date settings in the format:
                   {
                     "overall_date_range": {
                         "start": "2010-01",
                         "end": "2010-12"
                     },
                     "rolling_window": {
                         "train_window_months": 1,
                         "test_offset": 1
                     }
                   }
      table_name: Name of the database table to query (default: "data").

    Returns:
      tuple: (DataFrame containing the selected data, date_config dictionary)
             The SQL query filters rows where the date is between overall_date_range.start
             and overall_date_range.end.
    """
    try:
        # Build a comma-separated string of features.
        features_str = ", ".join(features)
        
        # Extract the overall date range from the configuration.
        overall_range = date_config.get("overall_date_range", {})
        overall_start = overall_range.get("start")
        overall_end = overall_range.get("end")
        
        query = (
            f"SELECT ticker, date, ret, {features_str} "
            f"FROM {table_name} "
            f"WHERE date BETWEEN '{overall_start}' AND '{overall_end}'"
        )
        logging.info(f"Constructed query: {query}")
        
        # Fetch the data using the constructed query.
        df = fetch_stock_data(engine, query)
        logging.info(f"Data fetched with date configuration: {date_config}")
        
        return df, date_config
    except Exception as e:
        logging.error(f"Error fetching stock data with features: {e}")
        raise
