import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "concealing baby swallow" w danych cenowych
def wykryj_concealing_baby_swallow(dane):
    """
    Wykrywa formację "concealing baby swallow" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    concealing_baby_swallow = np.full(n, False)
    for i in range(3, n):
        pierwsza_swieca = dane.iloc[i - 3]
        druga_swieca = dane.iloc[i - 2]
        trzecia_swieca = dane.iloc[i - 1]
        czwarta_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (długa, spadkowa, bez dolnego cienia)
        if pierwsza_swieca['close'] >= pierwsza_swieca['open'] or pierwsza_swieca['close'] != pierwsza_swieca['low']:
            continue  # Pierwsza świeca nie jest spadkowa lub ma dolny cień, pomijamy

        # 2. Sprawdzenie drugiej świecy (długa, spadkowa, bez dolnego cienia z luką w dół)
        if (druga_swieca['close'] >= druga_swieca['open'] or druga_swieca['close'] != druga_swieca['low'] or
                druga_swieca['open'] >= pierwsza_swieca['close']):
            continue  # Druga świeca nie jest spadkowa lub brak luki w dół, pomijamy

        # 3. Sprawdzenie trzeciej świecy (wzrostowa, w pełni zawarta w korpusie drugiej świecy)
        if not (trzecia_swieca['close'] > trzecia_swieca['open'] and
                trzecia_swieca['high'] <= druga_swieca['open'] and
                trzecia_swieca['low'] >= druga_swieca['close']):
            continue  # Trzecia świeca nie jest wzrostowa lub nie jest w pełni zawarta w drugiej świecy, pomijamy

        # 4. Sprawdzenie czwartej świecy (długa, spadkowa zamykająca się poniżej zamknięcia trzeciej świecy)
        if czwarta_swieca['close'] >= czwarta_swieca['open'] or czwarta_swieca['close'] >= trzecia_swieca['close']:
            continue  # Czwarta świeca nie jest spadkowa lub zamyka się powyżej trzeciej świecy, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        concealing_baby_swallow[i] = True

    return concealing_baby_swallow
