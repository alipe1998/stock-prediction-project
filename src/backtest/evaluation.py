# src/evaluate/evaluate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def calculate_cumulative_returns(returns_series):
    """
    Calculate cumulative (accumulated) returns from a series of portfolio returns.
    """
    cumulative_returns = (1 + returns_series).cumprod()
    return cumulative_returns

def calculate_drawdowns(cumulative_returns):
    """
    Calculate drawdowns from cumulative returns.
    """
    prior_peaks = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / prior_peaks - 1
    return drawdowns

def evaluate_portfolio(returns_data, risk_free_series=None):
    """
    Evaluates portfolio performance.
    
    Parameters:
      returns_data: Either a pd.Series of portfolio returns (indexed by date as a PeriodIndex or DatetimeIndex)
                    OR a DataFrame with columns 'month' and 'portfolio_return'.
      risk_free_series: pd.Series of risk-free rates (optional, must align with the returns_series index).
      
    Returns:
      metrics: Dictionary containing:
          - cumulative_returns: Series of accumulated returns.
          - drawdowns: Series of drawdowns.
          - mean_return: Mean monthly return.
          - std_dev: Standard deviation of monthly returns.
          - sharpe_ratio: Annualized Sharpe ratio.
    """
    # Convert DataFrame to Series if necessary.
    if isinstance(returns_data, pd.DataFrame):
        if 'month' in returns_data.columns and 'portfolio_return' in returns_data.columns:
            df = returns_data.copy()
            # Convert month to datetime (assumes format 'YYYY-MM')
            df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
            df.set_index('month', inplace=True)
            returns_series = df['portfolio_return']
        else:
            raise ValueError("DataFrame must contain columns 'month' and 'portfolio_return'.")
    elif isinstance(returns_data, pd.Series):
        returns_series = returns_data
    else:
        raise ValueError("returns_data must be either a DataFrame or a Series.")
    
    cumulative_returns = calculate_cumulative_returns(returns_series)
    drawdowns = calculate_drawdowns(cumulative_returns)
    
    mean_return = returns_series.mean()
    std_dev = returns_series.std()
    
    # Calculate excess returns if a risk-free series is provided; otherwise assume zero risk-free rate.
    if risk_free_series is not None:
        excess_returns = returns_series - risk_free_series
    else:
        excess_returns = returns_series
        
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(12)
    
    metrics = {
        "cumulative_returns": cumulative_returns,
        "drawdowns": drawdowns,
        "mean_return": mean_return,
        "std_dev": std_dev,
        "sharpe_ratio": sharpe_ratio
    }
    return metrics

def plot_performance(returns_data):
    """
    Plot cumulative returns and drawdowns on a shared chart.
    Accepts either a Series of returns or a DataFrame with 'month' and 'portfolio_return' columns.
    """
    # Convert input to Series if necessary.
    if isinstance(returns_data, pd.DataFrame):
        if 'month' in returns_data.columns and 'portfolio_return' in returns_data.columns:
            df = returns_data.copy()
            df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
            df.set_index('month', inplace=True)
            returns_series = df['portfolio_return']
        else:
            raise ValueError("DataFrame must contain columns 'month' and 'portfolio_return'.")
    elif isinstance(returns_data, pd.Series):
        returns_series = returns_data
    else:
        raise ValueError("returns_data must be either a DataFrame or a Series.")
    
    # Calculate cumulative returns and drawdowns.
    cumulative_returns = calculate_cumulative_returns(returns_series)
    drawdowns = calculate_drawdowns(cumulative_returns)
    
    # If the index is a PeriodIndex, convert it to DatetimeIndex.
    if isinstance(returns_series.index, pd.PeriodIndex):
        time_index = returns_series.index.to_timestamp()
    else:
        time_index = returns_series.index
    
    # Create figure and twin axes.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    # Plot cumulative returns on ax2 and drawdowns on ax1.
    ax2.plot(time_index, cumulative_returns, 'b-', label="Cumulative Returns")
    ax1.plot(time_index, drawdowns, 'r-', label="Drawdowns")
    
    # Set axis labels and title.
    ax1.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Returns', color='b')
    ax1.set_ylabel('Drawdowns', color='r')
    plt.title("Portfolio Performance")
    
    # Rotate the x-axis tick labels using the primary axes.
    ax1.tick_params(axis='x', rotation=45)
    # Alternatively, automatically format the x-date labels.
    fig.autofmt_xdate()
    
    # Enable grid lines on the primary y-axis.
    ax1.grid(True)
    
    # Adjust layout to prevent clipping of tick labels.
    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    # Example usage:
    # Assume you have a CSV with monthly portfolio returns, with columns 'month' and 'portfolio_return'
    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    rets_df = pd.read_csv(DATA_DIR / "rets2.csv")
    # The CSV should contain 'month' (format 'YYYY-MM') and 'portfolio_return'
    
    metrics = evaluate_portfolio(rets_df)
    cumulative_returns = metrics["cumulative_returns"]
    drawdowns = metrics["drawdowns"]
    
    print("Accumulation (cumulative returns):", cumulative_returns.tail(1).values[0])
    print("Maximum Drawdown:", drawdowns.min())
    print("Mean Return:", metrics["mean_return"])
    print("Standard Deviation:", metrics["std_dev"])
    print("Sharpe Ratio:", metrics["sharpe_ratio"])
    
    plot_performance(rets_df)
