import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from dotenv import load_dotenv
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Load API keys from environment variables
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

ALPACA_KEY = os.getenv("alpaca_key")
ALPACA_SECRET = os.getenv("alpaca_secret_key")

if not ALPACA_KEY or not ALPACA_SECRET:
    raise EnvironmentError("Missing Alpaca API keys in .env")

# Trading and Market Data clients
trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
market_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

def load_predictions(filepath):
    df = pd.read_csv(filepath)
    logging.info(f"Loaded predictions from {filepath}")
    return df

def get_account_value():
    account = trading_client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    logging.info(f"Account equity: ${equity:.2f}, Buying power: ${buying_power:.2f}")
    return equity, buying_power

def get_current_price(symbol):
    request_params = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
    latest_quote = market_client.get_stock_latest_quote(request_params)
    quote = latest_quote.get(symbol)
    if quote and quote.ask_price and quote.ask_price > 0:
        return quote.ask_price
    elif quote and quote.bid_price and quote.bid_price > 0:
        return quote.bid_price
    else:
        logging.warning(f"No valid ask/bid price found for {symbol}.")
        return None

def is_tradable(symbol, side):
    try:
        asset = trading_client.get_asset(symbol)
        if not asset.tradable:
            logging.warning(f"{symbol} is not tradable.")
            return False
        if side == OrderSide.SELL and not asset.shortable:
            logging.warning(f"{symbol} cannot be shorted.")
            return False
        return True
    except Exception as e:
        logging.warning(f"Asset check failed for {symbol}: {e}")
        return False

def submit_order(symbol, qty, side):
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order_data=order)
        logging.info(f"{side.value.capitalize()} order placed for {symbol}: {qty} shares.")
    except Exception as e:
        logging.error(f"Failed to place order for {symbol}: {e}")

def execute_portfolio_strategy(df, long_pct=1.45, short_pct=0.45):
    equity, buying_power = get_account_value()

    longs = df.sort_values("predicted_ret", ascending=False).head(10)
    shorts = df.sort_values("predicted_ret", ascending=True).head(5)

    long_capital = equity * long_pct
    short_capital = equity * short_pct

    # Allocate capital safely
    long_per_stock = long_capital / len(longs)
    short_per_stock = short_capital / len(shorts)

    # Track spent buying power
    total_spent = 0

    # Submit long orders
    for _, row in longs.iterrows():
        symbol = row['ticker']
        if is_tradable(symbol, OrderSide.BUY):
            price = get_current_price(symbol)
            if price:
                qty = int(long_per_stock / price)
                if qty > 0 and total_spent + qty * price <= buying_power:
                    submit_order(symbol, qty, OrderSide.BUY)
                    total_spent += qty * price
                else:
                    logging.warning(f"Skipping {symbol}: insufficient buying power or zero qty.")

    # Submit short orders
    for _, row in shorts.iterrows():
        symbol = row['ticker']
        if is_tradable(symbol, OrderSide.SELL):
            price = get_current_price(symbol)
            if price:
                qty = int(short_per_stock / price)
                if qty > 0 and total_spent + qty * price <= buying_power:
                    submit_order(symbol, qty, OrderSide.SELL)
                    total_spent += qty * price
                else:
                    logging.warning(f"Skipping short {symbol}: insufficient buying power or zero qty.")

if __name__ == "__main__":
    predictions_df = load_predictions()
    execute_portfolio_strategy(predictions_df)
