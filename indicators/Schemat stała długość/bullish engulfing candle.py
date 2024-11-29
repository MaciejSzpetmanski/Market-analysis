import pandas as pd
import numpy as np


# Funkcja wykrywająca formację "bullish engulfing candle"
def wykryj_bullish_engulfing(dane):
    """
    Wykrywa formację "bullish engulfing candle" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    bullish_engulfing = np.full(n, False)
    for i in range(1, n):
        pierwsza_swieca = dane.iloc[i - 1]
        druga_swieca = dane.iloc[i]

        # Sprawdzenie, czy pierwsza świeca jest spadkowa
        if pierwsza_swieca['close'] >= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest spadkowa, pomijamy

        # Sprawdzenie, czy druga świeca jest wzrostowa i pochłania korpus pierwszej świecy
        if (druga_swieca['close'] > druga_swieca['open'] and
            druga_swieca['open'] < pierwsza_swieca['close'] and
            druga_swieca['close'] > pierwsza_swieca['open']):
            continue

        # Sprawdzanie czy świeca pochłąniająca jest odpowiednio większa
        if (druga_swieca['close']-druga_swieca['open'] >= 1.7*(pierwsza_swieca['open']-pierwsza_swieca['close'])):
            bullish_engulfing[i] = True  # Wykryto formację "bullish engulfing candle"

    return bullish_engulfing