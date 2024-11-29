import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "bearish belt hold" w danych cenowych
def wykryj_bearish_belt_hold(dane):
    """
    Wykrywa formację "bearish belt hold" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    bearish_belt_hold = np.full(n, False)
    for i in range(n):
        swieca = dane.iloc[i]

        # Sprawdzenie, czy świeca jest spadkowa (open > close)
        if swieca['open'] <= swieca['close']:
            continue  # Świeca nie jest spadkowa, pomijamy

        # Sprawdzenie, czy świeca otwiera się na najwyższym poziomie (brak górnego cienia)
        if swieca['open'] != swieca['high']:
            continue  # Świeca nie otwiera się na maksimum, pomijamy

        # Sprawdzenie, czy świeca jest długa, czyli zamyka się znacznie niżej
        dlugosc_swiecy = swieca['open'] - swieca['close']
        srednia_zmiennosc = (swieca['high'] - swieca['low']) * 0.5  # Średnia zmienność świecowa
        if dlugosc_swiecy < srednia_zmiennosc:
            continue  # Świeca nie jest wystarczająco długa, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        bearish_belt_hold[i] = True

    return bearish_belt_hold
