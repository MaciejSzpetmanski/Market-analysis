import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Funkcja wykrywająca formację "triple top" w danych cenowych
def wykryj_potrojny_szczyt(dane, odleglosc_szczytow=5, tolerancja_szczytow=0.02):
    """
    Wykrywa formację "triple top" w dostarczonym DataFrame.
    
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

    # Szukanie formacji "triple top"
    for i in range(2, len(szczyty)):
        # Trzy kolejne szczyty
        pierwszy_szczyt = wartosci_szczytow.iloc[i - 2]
        drugi_szczyt = wartosci_szczytow.iloc[i - 1]
        trzeci_szczyt = wartosci_szczytow.iloc[i]
        
        # Sprawdzanie warunków dla formacji "triple top"
        szczyty_podobne = (
            abs(pierwszy_szczyt - drugi_szczyt) <= tolerancja_szczytow * pierwszy_szczyt and
            abs(drugi_szczyt - trzeci_szczyt) <= tolerancja_szczytow * drugi_szczyt
        )
        
        # Doliny między szczytami muszą być niższe od szczytów
        dolina_1 = ceny_wysokie.iloc[szczyty[i - 2]:szczyty[i - 1]].min()
        dolina_2 = ceny_wysokie.iloc[szczyty[i - 1]:szczyty[i]].min()
        doliny_nizej = dolina_1 < pierwszy_szczyt and dolina_2 < drugi_szczyt and dolina_1 < trzeci_szczyt and dolina_2 < trzeci_szczyt

        # Jeśli trzy szczyty są podobne i doliny są niżej, zwróć True
        if szczyty_podobne and doliny_nizej:
            return True

    # Jeśli nie wykryto formacji "triple top", zwróć False
    return False
