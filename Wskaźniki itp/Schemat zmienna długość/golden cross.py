import pandas as pd

# Funkcja obliczająca średnią kroczącą
def oblicz_srednia_kroczaca(dane, okres):
    return dane['close'].rolling(window=okres).mean()

# Funkcja wykrywająca formację "golden cross"
def wykryj_golden_cross(dane, okres_szybki=50, okres_wolny=200):
    """
    Wykrywa formację "golden cross" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    okres_szybki (int): Okres dla szybkiej średniej kroczącej.
    okres_wolny (int): Okres dla wolnej średniej kroczącej.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    dane['srednia_szybka'] = oblicz_srednia_kroczaca(dane, okres_szybki)
    dane['srednia_wolna'] = oblicz_srednia_kroczaca(dane, okres_wolny)

    for i in range(1, len(dane)):
        # Sprawdzenie, czy szybka średnia przecina wolną średnią od dołu
        if (dane['srednia_szybka'].iloc[i - 1] < dane['srednia_wolna'].iloc[i - 1] and
            dane['srednia_szybka'].iloc[i] > dane['srednia_wolna'].iloc[i]):
            return True  # Wykryto formację "golden cross"

    return False  # Jeśli nie wykryto formacji
