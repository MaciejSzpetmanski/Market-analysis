import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "bearish breakaway" w danych cenowych
def wykryj_bearish_breakaway(dane):
    """
    Wykrywa formację "bearish breakaway" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    bearish_breakaway = np.full(n, False)
    for i in range(4, n):
        pierwsza_swieca = dane.iloc[i - 4]
        druga_swieca = dane.iloc[i - 3]
        trzecia_swieca = dane.iloc[i - 2]
        czwarta_swieca = dane.iloc[i - 1]
        piata_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (długa, wzrostowa)
        if pierwsza_swieca['close'] <= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest wzrostowa, pomijamy
        dlugosc_pierwszej = pierwsza_swieca['close'] - pierwsza_swieca['open']

        # 2. Sprawdzenie drugiej świecy (wzrostowa z luką w górę)
        if druga_swieca['close'] <= druga_swieca['open'] or druga_swieca['open'] <= pierwsza_swieca['close']:
            continue  # Druga świeca nie jest wzrostowa lub brak luki w górę, pomijamy

        # 3. Sprawdzenie trzeciej świecy (wzrostowa z mniejszym korpusem)
        korpus_trzeciej = abs(trzecia_swieca['close'] - trzecia_swieca['open'])
        if trzecia_swieca['close'] <= trzecia_swieca['open'] or korpus_trzeciej >= dlugosc_pierwszej:
            continue  # Trzecia świeca nie jest wzrostowa lub jej korpus jest zbyt duży, pomijamy

        # 4. Sprawdzenie czwartej świecy (mała, spadkowa świeca)
        if czwarta_swieca['close'] >= czwarta_swieca['open']:
            continue  # Czwarta świeca nie jest spadkowa, pomijamy

        # 5. Sprawdzenie piątej świecy (długa, spadkowa świeca zamykająca się poniżej zamknięcia pierwszej świecy)
        if piata_swieca['close'] >= piata_swieca['open'] or piata_swieca['close'] >= pierwsza_swieca['close']:
            continue  # Piąta świeca nie jest spadkowa lub zamknięcie nie jest poniżej pierwszej świecy, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        bearish_breakaway[i] = True

    # Jeśli nie wykryto formacji, zwróć False
    return bearish_breakaway
