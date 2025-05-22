#!/usr/bin/env python3
###
# author: Benjamin J West
# DO NOT REMOVE THIS HEADER
# multi ticker momentum strategy
# This script implements a multi-ticker momentum strategy using Alpaca's API.
# It fetches historical data, performs backtesting, and executes trades based on signals.
# the script uses the Alpaca SDK for data retrieval and trade execution.
# It also includes logging for tracking the trading process.
# the process is as follows:
# 1. Load environment variables for Alpaca API credentials.
# 2. Fetch historical data for a given symbol using the Alpaca SDK.
# 3. Perform backtesting to find the optimal lookback period and calculate returns.
# 4. Determine trading signals based on the backtest results.
# 5. sort the results by best backtest score and filter for buy signals.
# 6. check currently inveseted tickers to see If a sell signal is detected 
# 7. If the price of selling is greater then the price that stock was bought at then place a limit order to sell.
# 8. check if there are 4 or more tickers already invested.
# 9. If less than 4 tickers are invested, find new tickers to buy.
# 10. Execute trades based on the signals and available cash split by 1/4th of the .
# 11. If a buy signal is detected, place a limit order to buy.
# 12. Log all actions and errors for future reference.
###

import os
import yfinance as yf
import logging
import concurrent.futures
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

# Alpaca SDK imports for historical data
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()

# Alpaca API credentials and endpoint
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # or live URL if you are trading live

# Initialize Alpaca REST API
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Setup logging
log_folder = 'log'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_file = os.path.join(log_folder, 'trading_log.txt')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------- Backtesting and Signal Functions ----------------

def fetch_alpaca_data(symbol, start_date, end_date, timeframe='1Day'):
    """
    Fetch historical bar data from Alpaca Market Data API using the official SDK.
    Returns a DataFrame with columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("Alpaca API credentials not set in environment variables.")
    tf_map = {
        '1Day': TimeFrame.Day,
        '1Min': TimeFrame.Minute,
        '5Min': TimeFrame(5, TimeFrame.Minute),
        '15Min': TimeFrame(15, TimeFrame.Minute),
    }
    tf = tf_map.get(timeframe, TimeFrame.Day)
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_date,
        end=end_date,
        feed='iex'
    )
    try:
        barset = client.get_stock_bars(request_params)
    except Exception as e:
        raise RuntimeError(f"Could not fetch data from Alpaca SDK: {e}")
    if hasattr(barset, 'df') and not barset.df.empty:
        df = barset.df.xs(symbol)
        df = df.reset_index().rename(columns={
            'timestamp': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df.set_index('Date', inplace=True)
        return df
    else:
        raise ValueError("No data returned from Alpaca for symbol: " + symbol)

def backtest(symbol, start_date, end_date, min_lookback=5, max_lookback=60):
    # Use Alpaca market data for backtesting
    data = fetch_alpaca_data(symbol, start_date, end_date)
    if data.empty:
        raise ValueError(f"No data found for symbol {symbol} between {start_date} and {end_date}")

    close_data = data['Close'].to_numpy()

    best_lookback = 0
    best_return = float('-inf')
    best_portfolio_values = []

    for lookback_period in range(min_lookback, max_lookback + 1):
        balance = 10000  # Starting balance
        shares = 0
        last_buy_price = 0
        portfolio_values = []

        for i in range(len(close_data)):
            if i >= lookback_period:
                momentum = close_data[i].item() - close_data[i - lookback_period].item()
                if momentum < 0 and shares == 0:
                    shares_to_buy = int(balance // close_data[i].item())
                    if shares_to_buy > 0:
                        balance -= shares_to_buy * close_data[i].item()
                        shares += shares_to_buy
                        last_buy_price = close_data[i].item()
                elif momentum > 0 and shares > 0 and close_data[i].item() > last_buy_price:
                    balance += shares * close_data[i].item()
                    shares = 0

            portfolio_value = balance + shares * close_data[i].item()
            portfolio_values.append(portfolio_value)

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        if total_return > best_return and portfolio_values[-1] > portfolio_values[0]:
            best_return = total_return
            best_lookback = lookback_period
            best_portfolio_values = portfolio_values

    return data, best_portfolio_values, best_lookback, best_return


def get_trading_signal(symbol, data, lookback_period):
    """
    Given historical data and a lookback period, determine the trading signal.
    Returns:
      1 for a BUY signal,
     -1 for a SELL signal,
      0 for HOLD.
    """
    if len(data) < lookback_period + 1:
        raise ValueError("Not enough data to calculate momentum.")

    # Convert the 'Close' series to a NumPy array
    close_data = data['Close'].to_numpy()
    
    # Calculate momentum using the last element and the element lookback_period+1 ago
    momentum = close_data[-1] - close_data[-lookback_period - 1]
    if momentum < 0:
        return 1  # BUY signal
    elif momentum > 0:
        return -1  # SELL signal
    else:
        return 0  # HOLD


def find_optimal_lookback(symbol):
    """Run the backtest for the past year and return the optimal lookback period."""
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    _, _, best_lookback, best_return = backtest(symbol, start_date, end_date)
    logging.info(f"{symbol} - Optimal lookback period: {best_lookback} days; Best return: {best_return:.2%}")
    return best_lookback

def evaluate_ticker(ticker, start_date, end_date, min_lookback=5, max_lookback=60):
    ticker_alpha = ''.join(filter(str.isalpha, ticker))
    try:
        data, _, best_lookback, best_return = backtest(ticker_alpha, start_date, end_date, min_lookback, max_lookback)
        signal = get_trading_signal(ticker_alpha, data, best_lookback)
        return (ticker_alpha, best_return, signal, best_lookback)
    except Exception as e:
        logging.error(f"Error evaluating {ticker_alpha}: {e}")
        return None

def round_price(price):
    """
    Round the price to the nearest $0.01 if >= 1.00; otherwise to the nearest $0.0001.
    """
    if price >= 1.00:
        rounded_price = float(Decimal(price).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    else:
        rounded_price = float(Decimal(price).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
    return rounded_price

# ---------------- Trade Execution Function ----------------

def execute_trade(symbol, data, lookback_period, max_invest_per_ticker):
    try:
        account = api.get_account()
    except Exception as e:
        logging.error(f"Error retrieving account info: {e}")
        return

    positions = api.list_positions()
    current_position = None
    for position in positions:
        if position.symbol == symbol:
            current_position = position
            break

    signal = get_trading_signal(symbol, data, lookback_period)

    try:
        latest_trade = api.get_latest_trade(symbol)
        current_price = float(latest_trade.price)
    except Exception as e:
        logging.error(f"Error getting latest trade for {symbol}: {e}")
        return

    buy_limit_price = round_price(current_price * 1.005)
    sell_limit_price = round_price(current_price * 0.995)

    if signal == 1 and current_position is None:
        cash = float(account.cash)
        max_buy_cash = min(max_invest_per_ticker, cash)
        shares_to_buy = int(max_buy_cash // buy_limit_price)
        if shares_to_buy > 0:
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=shares_to_buy,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=buy_limit_price
                )
                logging.info(f"Placed BUY limit order for {shares_to_buy} shares of {symbol} at {buy_limit_price}.")
            except Exception as e:
                logging.error(f"Failed to place BUY limit order for {symbol}: {e}")
        else:
            logging.info(f"Insufficient cash to buy {symbol} at {buy_limit_price}.")
    elif signal == -1 and current_position is not None:
        # Ensure cost_basis and qty are scalars
        try:
            if isinstance(current_position.avg_entry_price, (list, tuple)) or \
               isinstance(current_position.avg_entry_price, (np.ndarray,)):
                cost_basis = float(current_position.avg_entry_price.item())
            else:
                cost_basis = float(current_position.avg_entry_price)
                
            if isinstance(current_position.qty, (list, tuple)) or \
               isinstance(current_position.qty, (np.ndarray,)):
                qty = int(current_position.qty.item())
            else:
                qty = int(current_position.qty)
        except Exception as e:
            logging.error(f"Error extracting position details for {symbol}: {e}")
            return

        if sell_limit_price > cost_basis:
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='limit',
                    time_in_force='day',
                    limit_price=sell_limit_price
                )
                logging.info(f"Placed SELL limit order for {qty} shares of {symbol} at {sell_limit_price}.")
            except Exception as e:
                logging.error(f"Failed to place SELL limit order for {symbol}: {e}")
        else:
            logging.info(f"Holding {symbol}: Sell limit ({sell_limit_price}) is not above cost basis ({cost_basis}).")
    else:
        logging.info(f"No trade executed for {symbol} (signal: {signal}, position held: {current_position is not None}).")



# ---------------- Main Trading Logic ----------------

def main():
    # Log connection status
    logging.info("Attempting to connect to Alpaca API...")
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        logging.error("Alpaca API credentials not found in environment variables.")
        return

    try:
        account = api.get_account()
        logging.info(f"Connected to Alpaca API. Account status: {account.status}")
        logging.info(f"  - Buying power: ${float(account.buying_power):.2f}")
        logging.info(f"  - Cash: ${float(account.cash):.2f}")
        logging.info(f"  - Portfolio value: ${float(account.portfolio_value):.2f}")

        positions = api.list_positions()
        if positions:
            logging.info("Existing positions found:")
            for pos in positions:
                logging.info(f"  - {pos.symbol}: {pos.qty} shares @ Avg Entry ${float(pos.avg_entry_price):.2f}")
        else:
            logging.info("No positions currently held.")
    except Exception as e:
        logging.error(f"Failed to connect or retrieve account information: {e}")
        return

    # --- SELL LOGIC: Check each owned ticker for sell signal ---
    invested_symbols = set([p.symbol for p in api.list_positions()])
    for pos in api.list_positions():
        ticker = pos.symbol
        try:
            data = fetch_alpaca_data(ticker, datetime.now() - timedelta(days=365), datetime.now())
            optimal_lookback = find_optimal_lookback(ticker)
            signal = get_trading_signal(ticker, data, optimal_lookback)
            logging.info(f"Evaluated {ticker}: signal={signal}, optimal_lookback={optimal_lookback}")
            if signal == -1:
                logging.info(f"Sell signal detected for {ticker}. Executing SELL order...")
                execute_trade(ticker, data, optimal_lookback, 0)
            else:
                logging.info(f"Holding position in {ticker} (no sell signal).")
        except Exception as e:
            logging.error(f"Error processing sell decision for {ticker}: {e}")

    # --- BUY LOGIC: If less than 4 tickers invested, search for new tickers to buy ---
    max_tickers = 4
    num_invested = len(invested_symbols)
    if num_invested < max_tickers:
        tickers_file = "tested_tickers.txt"
        try:
            with open(tickers_file, "r") as f:
                tickers = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(tickers)} tickers from {tickers_file}.")
        except Exception as e:
            logging.error(f"Error reading tickers file {tickers_file}: {e}")
            return

        # Remove already invested tickers
        available_tickers = [t for t in tickers if t not in invested_symbols]
        logging.info(f"{len(available_tickers)} tickers available for new investment (not already held).")

        # Calculate allocation per new ticker
        num_to_invest = max_tickers - num_invested
        allocation_per_ticker = float(account.buying_power) * (1.0 / max_tickers)
        logging.info(f"Will attempt to invest in up to {num_to_invest} new tickers, allocating ${allocation_per_ticker:.2f} per ticker.")

        # Find up to num_to_invest tickers with a buy signal, sorted by best backtest score
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        results = []

        logging.info("Scanning tickers for potential BUY signals...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_ticker = {executor.submit(evaluate_ticker, ticker, start_date, end_date): ticker for ticker in available_tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                res = future.result()
                if res is not None:
                    logging.info(f"Backtest result for {res[0]}: best_return={res[1]:.2%}, signal={res[2]}, lookback={res[3]}")
                    results.append(res)
        # Each result is a tuple: (ticker, best_return, signal, optimal_lookback)

        # Filter to only those with a BUY signal (signal==1)
        buy_candidates = [r for r in results if r[2] == 1]
        if not buy_candidates:
            logging.info("No BUY candidates found from scan.")
        else:
            # Sort candidates by best return (highest potential) descending.
            buy_candidates.sort(key=lambda x: x[1], reverse=True)
            # Try to buy up to num_to_invest tickers, in order of best backtest score
            bought = 0
            for chosen_ticker, chosen_return, chosen_signal, chosen_lookback in buy_candidates:
                if bought >= num_to_invest:
                    break
                try:
                    data = fetch_alpaca_data(chosen_ticker, datetime.now() - timedelta(days=365), datetime.now())
                    confirm_signal = get_trading_signal(chosen_ticker, data, chosen_lookback)
                    logging.info(f"Confirming buy for {chosen_ticker}: signal={confirm_signal}, lookback={chosen_lookback}")
                    if confirm_signal == 1:
                        logging.info(f"Placing BUY for {chosen_ticker} (backtest return {chosen_return:.2%}, lookback {chosen_lookback})")
                        execute_trade(chosen_ticker, data, chosen_lookback, allocation_per_ticker)
                        bought += 1
                    else:
                        logging.info(f"Skipping {chosen_ticker}: no current buy signal on latest data.")
                except Exception as e:
                    logging.error(f"Error executing trade for {chosen_ticker}: {e}")
    else:
        logging.info("Already invested in 4 or more tickers. No new buys attempted.")

if __name__ == "__main__":
    main()
