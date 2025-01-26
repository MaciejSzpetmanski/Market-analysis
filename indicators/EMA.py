def ema(close, period):
    ema_values = close.ewm(span=period, adjust=False).mean()
    return ema_values
