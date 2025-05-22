import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Add dotenv support to load .env file
from dotenv import load_dotenv
load_dotenv()

# Alpaca SDK imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Alpaca API credentials (replace with your own)
ALPACA_API_KEY = os.getenv("LIVE_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("LIVE_ALPACA_SECRET_KEY")

def fetch_alpaca_data(symbol, start_date, end_date, timeframe='1Day'):
    """
    Fetch historical bar data from Alpaca Market Data API using the official SDK.
    Returns a DataFrame with columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    # Check credentials
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print(f"ERROR: Alpaca API ALPACA_API_KEY {ALPACA_API_KEY} or ALPACA_SECRET_KEY {ALPACA_SECRET_KEY} not set.")
        print("ERROR: Alpaca API credentials not set.")
        print("Set LIVE_ALPACA_API_KEY and LIVE_ALPACA_SECRET_KEY as environment variables.")
        raise SystemExit(1)
    # Map string timeframe to SDK TimeFrame
    tf_map = {
        '1Day': TimeFrame.Day,
        '1Min': TimeFrame.Minute,
        '5Min': TimeFrame(5, TimeFrame.Minute),
        '15Min': TimeFrame(15, TimeFrame.Minute),
        # Add more if needed
    }
    tf = tf_map.get(timeframe, TimeFrame.Day)

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_date,
        end=end_date,
        feed='iex'  # Use free IEX feed to avoid SIP subscription error
    )
    try:
        barset = client.get_stock_bars(request_params)
    except Exception as e:
        print("ERROR: Could not fetch data from Alpaca SDK:", e)
        print("If you see a SIP data error, set feed='iex' in StockBarsRequest for free data access.")
        print("Check your API credentials and data subscription.")
        raise SystemExit(1)

    # barset.df is a multi-index DataFrame (symbol, datetime)
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
    # Download historical data for the specified period using Alpaca
    data = fetch_alpaca_data(symbol, start_date, end_date)

    best_lookback = 0
    best_return = float('-inf')
    best_portfolio_values = []

    for lookback_period in range(min_lookback, max_lookback + 1):
        # Simulate trading
        balance = 1000  # Starting balance
        shares = 0
        last_buy_price = 0
        portfolio_values = []

        for i in range(len(data)):
            # Calculate momentum
            if i >= lookback_period:
                momentum = float(data['Close'].iloc[i]) - float(data['Close'].iloc[i - lookback_period])

                if momentum < 0 and shares == 0:  # Buy signal
                    buy_limit = data['Close'].iloc[i] * 1.005  # 0.5% above current price
                    shares_to_buy = int(balance // buy_limit)
                    if shares_to_buy > 0:
                        balance -= shares_to_buy * data['Close'].iloc[i]
                        shares += shares_to_buy
                        last_buy_price = buy_limit

                elif momentum > 0 and shares > 0:  # Sell signal
    
                    balance += shares * data['Close'].iloc[i]
                    shares = 0


            portfolio_value = balance + shares * data['Close'].iloc[i]
            portfolio_values.append(portfolio_value)

        total_return = (float(portfolio_values[-1]) - float(portfolio_values[0])) / float(portfolio_values[0])

        if total_return > best_return: #and float(portfolio_values[-1]) > float(portfolio_values[0]):
            best_return = total_return
            best_lookback = lookback_period
            best_portfolio_values = portfolio_values

    return data, best_portfolio_values, best_lookback, best_return

def visualize_backtest(symbol, start_date, end_date, min_lookback=5, max_lookback=60):
    # Perform backtest with optimization
    data, best_portfolio_values, best_lookback, best_return = backtest(symbol, start_date, end_date, min_lookback, max_lookback)
    best_portfolio_values = np.array(best_portfolio_values, dtype=float).ravel()
    # Create a DataFrame with the results
    results = pd.DataFrame({
        'Date': data.index,
        'Close': data['Close'].squeeze(),
        'Portfolio Value': best_portfolio_values
    })

    # Calculate buy and sell signals
    results['Signal'] = 0
    for i in range(best_lookback, len(results)):
        momentum = results['Close'].iloc[i] - results['Close'].iloc[i - best_lookback]
        if momentum < 0 and results['Signal'].iloc[i-1] >= 0:
            results.loc[results.index[i], 'Signal'] = 1  # Buy signal (at valleys)
        elif momentum > 0 and results['Signal'].iloc[i-1] <= 0:
            results.loc[results.index[i], 'Signal'] = -1  # Sell signal (at peaks)

    # Set up the plot
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")

    ax1 = plt.gca()
    # Plot stock price
    sns.lineplot(x='Date', y='Close', data=results, ax=ax1, color='blue', label='Stock Price')
    ax1.set_ylabel('Stock Price')
    ax1.set_title(f'{symbol} Stock Price and Optimized Portfolio Value')

    # Add buy and sell markers
    buy_signals = results[results['Signal'] == 1]
    sell_signals = results[results['Signal'] == -1]
    ax1.scatter(buy_signals['Date'], buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal')
    ax1.scatter(sell_signals['Date'], sell_signals['Close'], color='red', marker='v', s=100, label='Sell Signal')

    # Plot portfolio value on secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(x='Date', y='Portfolio Value', data=results, ax=ax2, color='green', label='Portfolio Value')
    ax2.set_ylabel('Portfolio Value')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Calculate and display performance metrics
    total_return = best_return * 100
    # Place text in the lower right, away from legend and axes
    plt.gcf().text(0.98, 0.02, f'Total Return: {total_return:.2f}%', fontsize=11, ha='right', va='bottom')
    plt.gcf().text(0.98, 0.07, f'Best Lookback Period: {best_lookback} days', fontsize=11, ha='right', va='bottom')

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('backtest_visualization.png')
    plt.show()

def main():
    symbol = 'SQQQ'
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=(565))

    visualize_backtest(symbol, start_date, end_date)

if __name__ == "__main__":
    main()


'''
Penny Stocks
BBD	
LU	   
IQ	   
RLX	   
PLUG
PTON
SOUN <-
CLOV
ACHR  
BLUE
NSAV
HLLK
GHAV
SINC <- good
'''