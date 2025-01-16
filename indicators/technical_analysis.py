def rsi(data, window=5):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(data, window=5, num_std_dev=2):
    sma = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return sma, upper_band, lower_band

def adx(high, low, close, window=5):
    # Krok 1: Oblicz True Range (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Krok 2: Oblicz Directional Movement (DM)
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    # Krok 3: Wygładź TR, DM+ i DM- przy użyciu średniej wykładniczej
    atr = true_range.rolling(window=window).mean()
    plus_dm_smoothed = pd.Series(plus_dm).rolling(window=window).mean()
    minus_dm_smoothed = pd.Series(minus_dm).rolling(window=window).mean()

    # Krok 4: Oblicz DI+ i DI-
    plus_di = 100 * (plus_dm_smoothed / atr)
    minus_di = 100 * (minus_dm_smoothed / atr)

    # Krok 5: Oblicz DX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    # Krok 6: Oblicz ADX
    adx = dx.rolling(window=window).mean()

    return adx, plus_di, minus_di


# TODO replace Nan with 0

import pandas as pd
import numpy as np

# Przykładowe dane (zastąp swoimi danymi)
data = pd.Series(np.random.randn(100).cumsum())

# Obliczanie SMA i RSI
# sma_20 = sma(data, window=20)
# rsi_14 = rsi(data, window=14)

# print("SMA:", sma_20.tail())
# print("RSI:", rsi_14.tail())
