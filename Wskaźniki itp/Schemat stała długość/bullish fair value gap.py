import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "bullish fair value gap"
def wykryj_bullish_fair_value_gap(dane):
    """
    Wykrywa formację "bullish fair value gap" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    bullish_fair_value_gap = np.full(n, False)
    for i in range(2, n):
        poprzednia_swieca = dane.iloc[i - 2]
        srodkowa_swieca = dane.iloc[i - 1]
        obecna_swieca = dane.iloc[i]

        # Sprawdzenie, czy srodkowa świeca jest wzrostowa z dużym korpusem
        korpus_srodkowej = abs(srodkowa_swieca['close'] - srodkowa_swieca['open'])
        wysokosc_srodkowej = srodkowa_swieca['high'] - srodkowa_swieca['low']
        if srodkowa_swieca['close'] <= srodkowa_swieca['open'] or korpus_srodkowej < 0.5 * wysokosc_srodkowej:
            continue  # Świeca środkowa nie jest wystarczająco wzrostowa, pomijamy

        # Sprawdzenie, czy jest luka między ceną maksymalną pierwszej świecy a ceną minimalną trzeciej świecy
        if obecna_swieca['low'] > poprzednia_swieca['high']:
            bullish_fair_value_gap[i] = True  # Wykryto "bullish fair value gap"

    return bullish_fair_value_gap