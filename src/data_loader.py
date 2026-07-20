import numpy as np
import pandas as pd
import yfinance as yf

def fetch_reit_data(tickers, start_date, end_date):
    """Fetch daily adjusted close prices for candidate REITs."""
    df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    df = df.dropna(axis=1, thresh=int(len(df) * 0.9)) # Drop if missing > 10%
    df = df.fillna(method='ffill').dropna() # Forward fill to prevent look-ahead
    returns = np.log(df / df.shift(1)).dropna()
    return df, returns