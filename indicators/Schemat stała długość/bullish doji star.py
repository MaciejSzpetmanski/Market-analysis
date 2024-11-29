import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "bullish doji star"
def wykryj_bullish_doji_star(dane, doji_procent=0.1):
    """
    Wykrywa formację "bullish doji star" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy typu doji.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    bullish_doji_star = np.full(n, False)
    for i in range(1, n):
        pierwsza_swieca = dane.iloc[i - 1]
        druga_swieca = dane.iloc[i]

        # Sprawdzenie, czy pierwsza świeca jest spadkowa
        if pierwsza_swieca['close'] >= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest spadkowa, pomijamy

        # Sprawdzenie, czy druga świeca jest doji (mały korpus)
        roznica_doji = abs(druga_swieca['close'] - druga_swieca['open'])
        zakres_doji = doji_procent * (druga_swieca['high'] - druga_swieca['low'])
        if roznica_doji > zakres_doji:
            continue  # Druga świeca nie spełnia warunku doji

        # Sprawdzenie, czy druga świeca otwiera się z luką w dół
        if druga_swieca['high'] >= pierwsza_swieca['low']:
            continue  # Brak luki w dół, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        bullish_doji_star[i] = True

    return bullish_doji_star
