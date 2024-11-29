import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Funkcja wykrywająca formację "triple bottom" w danych cenowych
def wykryj_potrojne_dno(dane, odleglosc_dolkow=5, tolerancja_dolkow=0.02):
    """
    Wykrywa formację "triple bottom" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    odleglosc_dolkow (int): Minimalna odległość między dołkami.
    tolerancja_dolkow (float): Tolerancja procentowa dla podobieństwa głębokości dołków.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    # Używamy ceny niskiej (low) do identyfikacji dołków
    ceny_niskie = dane['low']

    # Wykrywanie dołków (lokalnych minimów)
    dolki, _ = find_peaks(-ceny_niskie, distance=odleglosc_dolkow)
    wartosci_dolkow = ceny_niskie.iloc[dolki]

    # Szukanie formacji "triple bottom"
    for i in range(2, len(dolki)):
        # Głębokości trzech kolejnych dołków
        pierwszy_dol = wartosci_dolkow.iloc[i - 2]
        drugi_dol = wartosci_dolkow.iloc[i - 1]
        trzeci_dol = wartosci_dolkow.iloc[i]
        
        # Sprawdzanie warunków dla formacji "triple bottom"
        dolki_podobne = (
            abs(pierwszy_dol - drugi_dol) <= tolerancja_dolkow * pierwszy_dol and
            abs(drugi_dol - trzeci_dol) <= tolerancja_dolkow * drugi_dol
        )
        
        # Szczyty między dołkami muszą być wyższe niż dołki
        szczyt_1 = ceny_niskie.iloc[dolki[i - 2]:dolki[i - 1]].max()
        szczyt_2 = ceny_niskie.iloc[dolki[i - 1]:dolki[i]].max()
        szczyty_wyzsze = szczyt_1 > pierwszy_dol and szczyt_2 > drugi_dol and szczyt_1 > trzeci_dol and szczyt_2 > trzeci_dol

        # Jeśli trzy dołki są podobne głębokością i szczyty między nimi są wyżej, zwróć True
        if dolki_podobne and szczyty_wyzsze:
            return True

    # Jeśli nie wykryto formacji "triple bottom", zwróć False
    return False
