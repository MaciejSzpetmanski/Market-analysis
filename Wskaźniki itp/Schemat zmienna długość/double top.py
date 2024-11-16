import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Funkcja wykrywająca formację "double top" w danych cenowych
def wykryj_podwojny_szczyt(dane, odleglosc_szczytow=5, tolerancja_szczytow=0.02):
    """
    Wykrywa formację "double top" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    odleglosc_szczytow (int): Minimalna odległość między szczytami.
    tolerancja_szczytow (float): Tolerancja procentowa dla podobieństwa wysokości szczytów.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    # Używamy ceny wysokiej (high) do identyfikacji szczytów
    ceny_wysokie = dane['high']

    # Wykrywanie szczytów (lokalnych maksimów)
    szczyty, _ = find_peaks(ceny_wysokie, distance=odleglosc_szczytow)
    wartosci_szczytow = ceny_wysokie.iloc[szczyty]

    # Szukanie formacji "double top"
    for i in range(1, len(szczyty)):
        # Wysokości szczytów
        pierwszy_szczyt = wartosci_szczytow.iloc[i - 1]
        drugi_szczyt = wartosci_szczytow.iloc[i]
        
        # Warunki formacji "double top"
        szczyty_podobne = abs(pierwszy_szczyt - drugi_szczyt) <= tolerancja_szczytow * pierwszy_szczyt
        dolina_miedzy_szczytami = ceny_wysokie.iloc[szczyty[i - 1]:szczyty[i]].min()
        dolina_nizej = dolina_miedzy_szczytami < pierwszy_szczyt and dolina_miedzy_szczytami < drugi_szczyt

        # Jeśli oba szczyty są podobne wysokością i dolina między nimi jest niższa, zwróć True
        if szczyty_podobne and dolina_nizej:
            return True

    # Jeśli nie wykryto formacji "double top", zwróć False
    return False
