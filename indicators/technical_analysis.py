import pandas as pd
import numpy as np

def rsi(data, window=5):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    result_with_nans = 100 - (100 / (1 + rs))
    result = np.nan_to_num(result_with_nans, nan=50)
    return result

def macd(data, fast=5, slow=10, signal=5):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(data, window=5, num_std_dev=2):
    sma = data.rolling(window=window).mean()
    sma = np.nan_to_num(sma, nan=0)
    std_dev = data.rolling(window=window).std()
    std_dev = std_dev.bfill()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return sma, upper_band, lower_band

def adx(high, low, close, window=5):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    atr = true_range.rolling(window=window, min_periods=1).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=high.index).rolling(window=window, min_periods=1).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=low.index).rolling(window=window, min_periods=1).mean()

    plus_di = 100 * (plus_dm_smoothed / atr)
    minus_di = 100 * (minus_dm_smoothed / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_list = dx.rolling(window=window, min_periods=1).mean()

    adx_list = adx_list.bfill()
    plus_di = plus_di.bfill()
    minus_di = minus_di.bfill()

    return adx_list, plus_di, minus_di
