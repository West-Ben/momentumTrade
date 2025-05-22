# momentumTrade

# Momentum-Based Trading Bot

This repository contains two primary Python scripts:

- **`momentum.py`**: A fully automated multi-ticker momentum trading bot that interfaces with the Alpaca API to evaluate tickers, backtest their performance, and execute live trades on a paper account.
- **`backtest_vizualize.py`**: A backtesting and visualization utility used to analyze the momentum performance of individual tickers over a range of lookback periods and select candidates for trading.

---

## 🚀 Overview

The trading strategy implemented in `momentum.py` is based on **momentum over varying lookback periods**, optimized via backtesting. The bot:

1. Loads a list of potential tickers from `tested_tickers.txt`.
2. Evaluates each ticker’s performance over the past year.
3. Determines trading signals based on the best-performing lookback period.
4. Executes limit orders using Alpaca’s paper trading environment.
5. Automatically manages buy and sell logic to keep portfolio size under a defined max (default: 4 positions).

---

## 📁 File Descriptions

### `momentum.py`

> Fully automated trading logic powered by Alpaca’s API.

**Key Features:**

- Uses Alpaca’s SDK to fetch historical market data (`StockHistoricalDataClient`).
- Backtests multiple lookback periods (5–60 days) to find optimal momentum windows.
- Executes limit buy/sell orders based on recent signal direction and cost basis.
- Limits trades to 4 active positions to avoid overexposure.
- Logs all activities to `log/trading_log.txt`.

### `backtest_vizualize.py`

> Use this to analyze and visualize the historical performance of any ticker.

**Purpose:**

- Run offline backtests to visualize buy/sell signals and resulting portfolio growth.
- Determine which tickers have the strongest momentum characteristics for inclusion in `tested_tickers.txt`.

**Usage Example:**

```bash
python backtest_vizualize.py
This will backtest a hardcoded symbol (e.g., SQQQ), save a PNG plot (backtest_visualization.png), and print total return and optimal lookback period.

🛠 Setup Instructions
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set Up Environment Variables:

Create a .env file in the root directory with:

env
Copy
Edit
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
LIVE_ALPACA_API_KEY=your_live_key (optional for visualization)
LIVE_ALPACA_SECRET_KEY=your_live_secret (optional for visualization)
Create Ticker List:

Populate a tested_tickers.txt file with one symbol per line. Example:

nginx
Copy
Edit
SQQQ
PLUG
IQ
SOUN
SINC
Use backtest_vizualize.py to determine which ones demonstrate strong momentum patterns.

Run the Bot:

bash
Copy
Edit
python momentum.py
📈 How to Choose Tickers
Use backtest_vizualize.py to:

Analyze individual tickers.

View buy/sell momentum signals visually.

Confirm that the ticker produces strong returns with consistent signals.

Update tested_tickers.txt based on the results.

📄 Logging
All trading activity and error messages are written to:

bash
Copy
Edit
log/trading_log.txt
This includes:

Account status

Trade execution details

Signal evaluations

API connection errors

🔒 Disclaimer
This tool is intended for educational and paper trading purposes only. Trading involves significant risk and you are solely responsible for any financial decisions made using this software.

🧠 Author
Benjamin J. West
Developer, Strategist, and Author of this trading bot

📌 TODO
 Add unit tests for backtesting logic

 Enable configurable max portfolio size

 Implement trade confirmation via SMS/email
