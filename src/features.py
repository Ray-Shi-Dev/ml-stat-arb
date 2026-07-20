import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def causal_savgol_filter(series, window_size=11, polyorder=3):
    """Causal Savitzky-Golay avoiding look-ahead bias using a rolling apply."""
    return series.rolling(window_size).apply(
        lambda x: savgol_filter(x, window_size, polyorder)[-1], raw=True
    )

def engineer_features(spread_series, horizon=5):
    """Generate ML features and the breakdown anomaly target."""
    df = pd.DataFrame({'Spread': spread_series})
    
    df = df.rolling(window=21).mean()
    df = df.rolling(window=21).std()
    
    df = df.shift(-horizon)
    df['Anomaly'] = np.where(
        abs(df - df) > (3 * df), 1, 0
    )
    
    df = causal_savgol_filter(df, window_size=11, polyorder=3)
    df = df - df
    df['Lag_1'] = df - df.shift(1)
    df['Lag_5'] = df - df.shift(5)
    df = (df + 1) / (df.shift(5) + 1)
    
    return df.dropna()