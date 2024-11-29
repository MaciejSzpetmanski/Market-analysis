import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "advance block" w danych cenowych
def ema(close, period):
    ema_values = close.ewm(span=period, adjust=False).mean()
    return ema_values
