import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Funkcja wykrywająca formację "Head and Shoulders" w danych cenowych
def wykryj_glowe_i_ramiona(dane, odleglosc_szczytow=5, tolerancja_ramion=0.05, minimalna_wysokosc_glowy=1.05):
    """
    Wykrywa formację "Head and Shoulders" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    odleglosc_szczytow (int): Minimalna odległość między szczytami.
    tolerancja_ramion (float): Tolerancja dla podobieństwa wysokości ramion.
    minimalna_wysokosc_glowy (float): Minimalny stosunek wysokości głowy do ramion.
    
    Zwraca:
    bool: True, jeśli wykryto schemat, False w przeciwnym wypadku.
    """
    # Wykorzystujemy ceny zamknięcia do wykrywania szczytów
    ceny_zamkniecia = dane['close']

    # Wykrywanie lokalnych szczytów
    szczyty, _ = find_peaks(ceny_zamkniecia, distance=odleglosc_szczytow)
    wartosci_szczytow = ceny_zamkniecia.iloc[szczyty]

    # Iteracja przez szczyty, aby znaleźć formację "Head and Shoulders"
    for i in range(1, len(szczyty) - 1):
        lewe_ramie = wartosci_szczytow.iloc[i - 1]
        glowa = wartosci_szczytow.iloc[i]
        prawe_ramie = wartosci_szczytow.iloc[i + 1]
        
        # Sprawdzanie warunków formacji
        ramiona_podobne = abs(lewe_ramie - prawe_ramie) <= tolerancja_ramion * lewe_ramie
        glowa_wyzsza = glowa > lewe_ramie * minimalna_wysokosc_glowy and glowa > prawe_ramie * minimalna_wysokosc_glowy

        # Jeśli spełnione są warunki na formację "Head and Shoulders", zwróć True
        if ramiona_podobne and glowa_wyzsza:
            return True

    # Jeśli żaden wzór nie spełnia warunków "Head and Shoulders", zwróć False
    return False
