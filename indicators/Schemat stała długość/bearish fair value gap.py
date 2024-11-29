import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "bearish fair value gap"
def wykryj_bearish_fair_value_gap(dane):
    """
    Wykrywa formację "bearish fair value gap" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    bearish_engulfing = np.full(n, False)
    for i in range(2, n):
        poprzednia_swieca = dane.iloc[i - 2]
        srodkowa_swieca = dane.iloc[i - 1]
        obecna_swieca = dane.iloc[i]

        # Sprawdzenie, czy srodkowa świeca jest spadkowa z dużym korpusem
        korpus_srodkowej = abs(srodkowa_swieca['close'] - srodkowa_swieca['open'])
        wysokosc_srodkowej = srodkowa_swieca['high'] - srodkowa_swieca['low']
        if srodkowa_swieca['close'] >= srodkowa_swieca['open'] or korpus_srodkowej < 0.5 * wysokosc_srodkowej:
            continue  # Świeca środkowa nie jest wystarczająco spadkowa, pomijamy

        # Sprawdzenie, czy jest luka między ceną minimalną pierwszej świecy a ceną maksymalną trzeciej świecy
        if obecna_swieca['high'] < poprzednia_swieca['low']:
            bearish_engulfing[i] = True  # Wykryto "bearish fair value gap"

    return bearish_engulfing