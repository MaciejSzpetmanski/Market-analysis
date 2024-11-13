import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "evening star"
def wykryj_evening_star(dane, doji_procent=0.1):
    """
    Wykrywa formację "evening star" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy o małym korpusie.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    evening_star = np.full(n, False)
    for i in range(2, n):
        pierwsza_swieca = dane.iloc[i - 2]
        druga_swieca = dane.iloc[i - 1]
        trzecia_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (wzrostowa)
        if pierwsza_swieca['close'] <= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest wzrostowa, pomijamy

        # 2. Sprawdzenie drugiej świecy (mały korpus, potencjalne doji)
        roznica_doja = abs(druga_swieca['close'] - druga_swieca['open'])
        zakres_doja = doji_procent * (druga_swieca['high'] - druga_swieca['low'])
        if roznica_doja > zakres_doja:
            continue  # Druga świeca nie jest doji lub ma mały korpus, pomijamy

        # 3. Sprawdzenie trzeciej świecy (spadkowa, zamykająca się poniżej środka korpusu pierwszej świecy)
        polowa_korpusu_pierwszej = pierwsza_swieca['open'] + 0.5 * (pierwsza_swieca['close'] - pierwsza_swieca['open'])
        if trzecia_swieca['close'] >= trzecia_swieca['open'] or trzecia_swieca['close'] >= polowa_korpusu_pierwszej:
            continue  # Trzecia świeca nie jest spadkowa lub zamyka się powyżej środka pierwszej świecy, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        evening_star[i] = True

    return evening_star