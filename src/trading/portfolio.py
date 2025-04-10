# src/portfolio/portfolio.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def construct_portfolio(predictions, long_count=150, short_count=50):
    """
    Constructs a portfolio based on predicted returns.
    
    Parameters:
      predictions: List of dictionaries with keys 'ticker', 'actual_ret', 'predicted_ret'.
      long_count: Number of stocks to long.
      short_count: Number of stocks to short.
      
    Returns:
      portfolio: DataFrame with tickers, predicted and actual returns, and assigned weights.
    """
    df = pd.DataFrame(predictions)
    if df.empty:
        logging.warning("No predictions provided to construct portfolio.")
        return pd.DataFrame()
    
    # Select the top 'long_count' stocks for long positions (highest predicted returns)
    longs = df.sort_values(by="predicted_ret", ascending=False).head(long_count).copy()
    # Select the bottom 'short_count' stocks for short positions (lowest predicted returns)
    shorts = df.sort_values(by="predicted_ret", ascending=True).head(short_count).copy()
    
    longs["weight"] = 1.0 / long_count
    shorts["weight"] = -1.0 / short_count
    
    portfolio = pd.concat([longs, shorts])
    logging.info(f"Constructed portfolio with {len(longs)} long and {len(shorts)} short positions.")
    return portfolio

def evaluate_portfolio_returns(results_df, long_count=150, short_count=50):
    """
    Constructs a portfolio for each test month in the backtest results and calculates the portfolio's return.
    
    For each month:
      - Build the portfolio using the predictions.
      - Compute the portfolio return as the weighted sum of actual returns.
    
    Parameters:
      results_df: DataFrame returned by run_backtest containing a 'test_month' column and a 'predictions' column.
      long_count: Number of stocks to long.
      short_count: Number of stocks to short.
      
    Returns:
      A DataFrame with two columns:
        - 'month': the test month.
        - 'portfolio_return': the weighted return for the portfolio in that month.
    """
    portfolio_returns = []
    
    for _, row in results_df.iterrows():
        month = row['test_month']
        predictions = row['predictions']
        portfolio_df = construct_portfolio(predictions, long_count=long_count, short_count=short_count)
        if portfolio_df.empty:
            logging.warning(f"No portfolio constructed for month {month}.")
            continue
        
        # Calculate portfolio return: weighted sum of actual returns.
        port_return = (portfolio_df['weight'] * portfolio_df['actual_ret']).sum()
        portfolio_returns.append({
            "month": month,
            "portfolio_return": port_return
        })
    
    return pd.DataFrame(portfolio_returns)
