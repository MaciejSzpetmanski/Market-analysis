import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "dark cloud cover" w danych cenowych
def wykryj_dark_cloud_cover(dane):
    """
    Wykrywa formację "dark cloud cover" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    dark_cloud_cover = np.full(n, False)
    for i in range(1, n):
        pierwsza_swieca = dane.iloc[i - 1]
        druga_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (długa, wzrostowa)
        if pierwsza_swieca['close'] <= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest wzrostowa, pomijamy

        # 2. Sprawdzenie drugiej świecy (spadkowa, otwierająca się powyżej zamknięcia pierwszej świecy)
        if druga_swieca['close'] >= druga_swieca['open'] or druga_swieca['open'] <= pierwsza_swieca['close']:
            continue  # Druga świeca nie jest spadkowa lub nie otwiera się powyżej zamknięcia pierwszej świecy, pomijamy

        # 3. Sprawdzenie, czy druga świeca zamyka się poniżej połowy korpusu pierwszej świecy
        polowa_korpusu_pierwszej = pierwsza_swieca['open'] + 0.5 * (pierwsza_swieca['close'] - pierwsza_swieca['open'])
        if druga_swieca['close'] >= polowa_korpusu_pierwszej:
            continue  # Zamknięcie drugiej świecy nie jest poniżej połowy korpusu pierwszej świecy, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        dark_cloud_cover[i] = True

    return dark_cloud_cover
