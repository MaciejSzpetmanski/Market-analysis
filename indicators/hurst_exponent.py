import numpy as np

def hurst_exponent(ts, max_lag=20):
    """
    Calculate the Hurst Exponent of a time series.
    
    Parameters:
        ts (array-like): The time series data (1D array).
        max_lag (int): Maximum lag to consider for the analysis.
    
    Returns:
        float: Estimated Hurst Exponent.
    """
    lags = range(2, max_lag + 1)
    tau = []

    for lag in lags:
        # Calculate lagged differences
        diff = np.diff(ts, n=lag)
        # Calculate standard deviation of differences
        tau.append(np.std(diff))

    # Fit a line in log-log space to estimate H
    log_lags = np.log(lags)
    log_tau = np.log(tau)
    slope, _ = np.polyfit(log_lags, log_tau, 1)

    # The slope corresponds to the Hurst Exponent
    return slope