import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "ascending doji star"
def wykryj_ascending_doji_star(dane, doji_procent=0.1):
    """
    Wykrywa formację "ascending doji star" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy typu doji.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    ascending_doji_star = np.full(n, False)
    for i in range(2, n):
        pierwsza_swieca = dane.iloc[i - 2]
        druga_swieca = dane.iloc[i - 1]
        trzecia_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (długa, spadkowa)
        if pierwsza_swieca['close'] >= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest spadkowa, pomijamy

        # 2. Sprawdzenie drugiej świecy (doji z małym korpusem)
        roznica_doji = abs(druga_swieca['close'] - druga_swieca['open'])
        zakres_doji = doji_procent * (druga_swieca['high'] - druga_swieca['low'])
        if roznica_doji > zakres_doji:
            continue  # Druga świeca nie jest doji, pomijamy

        # 3. Sprawdzenie trzeciej świecy (wzrostowa świeca, zamknięcie powyżej zamknięcia pierwszej świecy)
        if trzecia_swieca['close'] <= trzecia_swieca['open'] or trzecia_swieca['close'] <= pierwsza_swieca['close']:
            continue  # Trzecia świeca nie jest wzrostowa lub nie zamyka się powyżej zamknięcia pierwszej świecy, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        ascending_doji_star[i] = True

    return ascending_doji_star
