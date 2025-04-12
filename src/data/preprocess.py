# src/data/preprocess.py
import pandas as pd
import os
import logging
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'

logging.basicConfig(level=logging.INFO)

def drop_na(df, **kwargs):
    return df.dropna()

def filter_price(df, price_col="price", min_price=5, **kwargs):
    if price_col not in df.columns:
        logging.warning(f"Column {price_col} not found. Skipping price filter.")
        return df
    filtered_df = df[df[price_col] >= min_price]
    logging.info(f"Filtered data: {len(df) - len(filtered_df)} rows removed with price below {min_price}.")
    return filtered_df

def quantile_transform(df, features, **kwargs):
    qt = QuantileTransformer(output_distribution="normal")
    try:
        # Make a copy to avoid modifying a slice
        df = df.copy()
        # Ensure the feature columns are floats
        df[features] = df[features].astype(float)
        # Fit and transform the features
        transformed = qt.fit_transform(df[features])
        # Create a DataFrame with the transformed values, preserving index and column names
        transformed_df = pd.DataFrame(transformed, index=df.index, columns=features)
        # Reassign the transformed DataFrame to the original DataFrame columns
        df[features] = transformed_df
        return df
    except Exception as e:
        logging.error(f"Quantile transformation failed: {e}")
        raise


def polynomial_features(df, features, degree=2, **kwargs):
    try:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_feats = poly.fit_transform(df[features])
        poly_feature_names = poly.get_feature_names_out(features)
        df_poly = pd.DataFrame(poly_feats, columns=poly_feature_names, index=df.index)
        # Optionally drop original features or keep them; here we keep both
        df = pd.concat([df, df_poly], axis=1)
        logging.info("Polynomial features added.")
        return df
    except Exception as e:
        logging.error(f"Polynomial features generation failed: {e}")
        raise

def encode_industry(df, **kwargs):
    """
    Merge industry information into the DataFrame based on SIC code ranges and apply one-hot encoding.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing at least a 'siccd' column.
      industry_file (str): Path to the CSV file with SIC ranges and industry labels.
      
    Returns:
      pd.DataFrame: A new DataFrame with the industry information added and one-hot encoded.
    """
    try:
        df = df.copy()
        industry_file = DATA_DIR / 'raw' / 'siccodes12.csv'
        # Check if the industry file exists
        if not os.path.exists(industry_file):
            logging.error(f"Industry file {industry_file} not found.")
            return df

        # Read the industry CSV.
        # The CSV should have columns: start, end, and industry.
        industry_df = pd.read_csv(industry_file)
        # Ensure that the "start" and "end" columns are treated as integers.
        industry_df['start'] = industry_df['start'].astype(int)
        industry_df['end'] = industry_df['end'].astype(int)
        
        # Define a helper function to lookup industry from a SIC code.
        def lookup_industry(sic):
            try:
                # Attempt to convert the sic code to an integer.
                sic = int(sic)
            except (ValueError, TypeError):
                return "Other"
            # Find the industry whose range covers the given sic value.
            match = industry_df[(industry_df['start'] <= sic) & (sic <= industry_df['end'])]
            if not match.empty:
                # Return the first matching industry's name.
                return match.iloc[0]['industry']
            else:
                return "Other"
        
        # Apply the lookup function to the "siccd" column
        df['industry'] = df['siccd'].apply(lookup_industry)
        
        # Check if the industry column was successfully created.
        if "industry" in df.columns:
            # One-hot encode the industry column with a custom prefix if desired.
            df = pd.get_dummies(df, columns=["industry"], prefix="ind")
        else:
            logging.warning("No industry column found after applying SIC code lookup; skipping encoding.")
            
        return df

    except Exception as e:
        logging.error(f"Industry encoding failed: {e}")
        raise


def preprocess_data(df, features, pipeline_steps):
    """
    Preprocess the data by applying a sequence of steps provided in pipeline_steps.
    
    Parameters:
    - df: raw DataFrame.
    - features: list of features to use (for quantile transform, polynomial features, etc.).
    - pipeline_steps: a list of strings indicating which preprocessing steps to apply.
    
    Returns:
    - Preprocessed DataFrame.
    """
    try:
        logging.info("Starting preprocessing pipeline.")
        # Apply each step in order.
        for step in pipeline_steps:
            if step == "drop_na":
                df = drop_na(df)
            elif step == "filter_price":
                df = filter_price(df)
            elif step == "quantile_transform":
                df = quantile_transform(df, features)
            elif step == "polynomial_features":
                df = polynomial_features(df, features)
            elif step == "encode_industry":
                df = encode_industry(df)
            else:
                logging.warning(f"Preprocessing step {step} not recognized. Skipping.")
        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        raise
