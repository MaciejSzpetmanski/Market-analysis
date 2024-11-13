import pandas as pd

# Funkcja wykrywająca formację "hammer"
def wykryj_hammer(dane):
    """
    Wykrywa formację "hammer" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(len(dane)):
        swieca = dane.iloc[i]

        # Obliczenie wielkości korpusu, górnego i dolnego cienia
        korpus = abs(swieca['close'] - swieca['open'])
        dolny_cien = swieca['open'] - swieca['low'] if swieca['close'] > swieca['open'] else swieca['close'] - swieca['low']
        gorny_cien = swieca['high'] - swieca['close'] if swieca['close'] > swieca['open'] else swieca['high'] - swieca['open']

        # Warunki formacji hammer: długi dolny cień, mały korpus i brak górnego cienia
        if dolny_cien >= 2 * korpus and gorny_cien <= 0.2 * korpus:
            return True  # Wykryto formację "hammer"

    return False  # Jeśli nie wykryto formacji

