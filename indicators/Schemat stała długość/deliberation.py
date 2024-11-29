import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "deliberation" w danych cenowych
def wykryj_deliberation(dane):
    """
    Wykrywa formację "deliberation" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    deliberation = np.full(n, False)
    for i in range(2, n):
        pierwsza_swieca = dane.iloc[i - 2]
        druga_swieca = dane.iloc[i - 1]
        trzecia_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (długa, wzrostowa)
        if pierwsza_swieca['close'] <= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest wzrostowa, pomijamy

        # 2. Sprawdzenie drugiej świecy (wzrostowa, podobna długością do pierwszej)
        korpus_pierwszej = pierwsza_swieca['close'] - pierwsza_swieca['open']
        korpus_drugiej = druga_swieca['close'] - druga_swieca['open']
        if druga_swieca['close'] <= druga_swieca['open'] or not (0.8 * korpus_pierwszej <= korpus_drugiej <= 1.2 * korpus_pierwszej):
            continue  # Druga świeca nie jest wzrostowa lub nie jest podobnej długości, pomijamy

        # 3. Sprawdzenie trzeciej świecy (wzrostowa, mniejszy korpus, może tworzyć lukę)
        korpus_trzeciej = trzecia_swieca['close'] - trzecia_swieca['open']
        if trzecia_swieca['close'] <= trzecia_swieca['open'] or korpus_trzeciej >= korpus_drugiej:
            continue  # Trzecia świeca nie jest wzrostowa lub ma zbyt duży korpus, pomijamy
        
        # Opcjonalne sprawdzenie: trzecia świeca otwiera się blisko zamknięcia drugiej lub z luką
        if not (trzecia_swieca['open'] >= druga_swieca['close'] or abs(trzecia_swieca['open'] - druga_swieca['close']) <= 0.1 * korpus_drugiej):
            continue  # Brak sygnału osłabienia, pomijamy
        
        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        deliberation[i] = True

    return deliberation
