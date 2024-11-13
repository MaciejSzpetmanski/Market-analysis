import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "bearish engulfing candle"
def wykryj_bearish_engulfing(dane):
    """
    Wykrywa formację "bearish engulfing candle" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    bearish_engulfing = np.full(n, False)
    for i in range(1, n):
        pierwsza_swieca = dane.iloc[i - 1]
        druga_swieca = dane.iloc[i]

        # Sprawdzenie, czy pierwsza świeca jest wzrostowa
        if pierwsza_swieca['close'] <= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest wzrostowa, pomijamy

        # Sprawdzenie, czy druga świeca jest spadkowa i pochłania korpus pierwszej świecy
        if (druga_swieca['close'] < druga_swieca['open'] and
            druga_swieca['open'] > pierwsza_swieca['close'] and
            druga_swieca['close'] < pierwsza_swieca['open']):
            continue

        # Sprawdzanie czy świeca pochłąniająca jest odpowiednio większa
        if (druga_swieca['open']-druga_swieca['close'] >= 1.7*(pierwsza_swieca['close']-pierwsza_swieca['open'])):
            bearish_engulfing[i] = True  # Wykryto formację "bearish engulfing candle"

    return bearish_engulfing