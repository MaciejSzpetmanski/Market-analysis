import pandas as pd

# Funkcja wykrywająca formację "bullish pipe formation"
def wykryj_bullish_pipe_formation(dane, minimalny_korpus=0.8):
    """
    Wykrywa formację "bullish pipe formation" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    minimalny_korpus (float): Minimalny stosunek korpusu świecy do jej wysokości.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(1, len(dane)):
        pierwsza_swieca = dane.iloc[i - 1]
        druga_swieca = dane.iloc[i]

        # Sprawdzenie, czy pierwsza świeca jest długą świecą spadkową
        korpus_pierwszej = abs(pierwsza_swieca['close'] - pierwsza_swieca['open'])
        wysokosc_pierwszej = pierwsza_swieca['high'] - pierwsza_swieca['low']
        if pierwsza_swieca['close'] >= pierwsza_swieca['open'] or korpus_pierwszej < minimalny_korpus * wysokosc_pierwszej:
            continue  # Pierwsza świeca nie jest wystarczająco długa i spadkowa

        # Sprawdzenie, czy druga świeca jest długą świecą wzrostową
        korpus_drugiej = abs(druga_swieca['close'] - druga_swieca['open'])
        wysokosc_drugiej = druga_swieca['high'] - druga_swieca['low']
        if druga_swieca['close'] <= druga_swieca['open'] or korpus_drugiej < minimalny_korpus * wysokosc_drugiej:
            continue  # Druga świeca nie jest wystarczająco długa i wzrostowa

        # Sprawdzenie, czy druga świeca zamyka się powyżej zamknięcia pierwszej świecy
        if druga_swieca['close'] > pierwsza_swieca['close']:
            return True  # Wykryto "bullish pipe formation"

    return False  # Jeśli nie wykryto formacji
