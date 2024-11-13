import pandas as pd

# Funkcja wykrywająca formację "bearish spike"
def wykryj_bearish_spike(dane, stosunek_cienia_do_korpusu=2):
    """
    Wykrywa formację "bearish spike" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    stosunek_cienia_do_korpusu (float): Minimalny stosunek górnego cienia do korpusu.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(len(dane)):
        swieca = dane.iloc[i]

        # Obliczenie wielkości korpusu, górnego i dolnego cienia
        korpus = abs(swieca['close'] - swieca['open'])
        dolny_cien = swieca['open'] - swieca['low'] if swieca['close'] > swieca['open'] else swieca['close'] - swieca['low']
        gorny_cien = swieca['high'] - swieca['close'] if swieca['close'] > swieca['open'] else swieca['high'] - swieca['open']

        # Warunki formacji bearish spike: długi górny cień i mały korpus oraz brak dolnego cienia
        if gorny_cien >= stosunek_cienia_do_korpusu * korpus and dolny_cien <= 0.2 * gorny_cien:
            return True  # Wykryto formację "bearish spike"

    return False  # Jeśli nie wykryto formacji
