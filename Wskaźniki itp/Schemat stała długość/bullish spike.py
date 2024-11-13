import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "bullish spike"
def wykryj_bullish_spike(dane, stosunek_cienia_do_korpusu=2):
    """
    Wykrywa formację "bullish spike" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    stosunek_cienia_do_korpusu (float): Minimalny stosunek dolnego cienia do korpusu.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    bullish_spike = np.full(n, False)
    for i in range(n):
        swieca = dane.iloc[i]

        # Obliczenie wielkości korpusu, górnego i dolnego cienia
        korpus = abs(swieca['close'] - swieca['open'])
        dolny_cien = swieca['open'] - swieca['low'] if swieca['close'] > swieca['open'] else swieca['close'] - swieca['low']
        gorny_cien = swieca['high'] - swieca['close'] if swieca['close'] > swieca['open'] else swieca['high'] - swieca['open']

        # Warunki formacji bullish spike: długi dolny cień i mały korpus oraz brak górnego cienia
        if dolny_cien >= stosunek_cienia_do_korpusu * korpus and gorny_cien <= 0.2 * dolny_cien:
            bullish_spike[i] = True  # Wykryto formację "bullish spike"

    return bullish_spike