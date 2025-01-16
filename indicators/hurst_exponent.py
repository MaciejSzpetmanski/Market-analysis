import numpy as np

# def hurst(data, max_lag=10):
#     """
#     source: https://www.linkedin.com/pulse/rolling-hurst-exponent-python-trading-jakub-polec-e92yf
#     Calculates the Hurst Exponent using the Rescaled Range (R/S) analysis method.
#     """
#     # requires max_lag + 3 observations
#     if len(data) < max_lag + 3:
#         return 0.5
#         # return 0.5, 1.5

#     log_returns = np.diff(np.log(data))
#     # lags = range(2, max_lag + 1)
#     # tau = [np.sqrt(np.std(np.subtract(log_returns[lag:], log_returns[:-lag]))) for lag in lags]
    
#     # poly = np.polyfit(np.log(lags), np.log(tau), 1)
#     # hurst_exponent = poly[0]*2.0
#     # fractal_dimension = 2 - hurst_exponent
    
    
#     lags = range(2, max_lag + 1)
    
#     tau = []
#     for lag in lags:
#         tau_lag = np.sqrt(np.std(np.subtract(log_returns[lag:], log_returns[:-lag])))
#         tau.append(tau_lag)
    
#     poly = np.polyfit(np.log(lags), np.log(tau), 1)
#     hurst_exponent = poly[0] * 2.0
    
#     return hurst_exponent
