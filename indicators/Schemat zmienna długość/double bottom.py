import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Funkcja wykrywająca formację "double bottom" w danych cenowych
def wykryj_podwojne_dno(dane, odleglosc_dolkow=5, tolerancja_dolkow=0.02):
    """
    Wykrywa formację "double bottom" w dostarczonym DataFrame.
    
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

    # Szukanie formacji "double bottom"
    for i in range(1, len(dolki)):
        # Głębokości dwóch kolejnych dołków
        pierwszy_dol = wartosci_dolkow.iloc[i - 1]
        drugi_dol = wartosci_dolkow.iloc[i]
        
        # Warunki formacji "double bottom"
        dolki_podobne = abs(pierwszy_dol - drugi_dol) <= tolerancja_dolkow * pierwszy_dol
        szczyt_miedzy_dolkami = ceny_niskie.iloc[dolki[i - 1]:dolki[i]].max()
        szczyt_wyzszy = szczyt_miedzy_dolkami > pierwszy_dol and szczyt_miedzy_dolkami > drugi_dol

        # Jeśli oba dołki są podobne głębokością i szczyt między nimi jest wyżej, zwróć True
        if dolki_podobne and szczyt_wyzszy:
            return True

    # Jeśli nie wykryto formacji "double bottom", zwróć False
    return False
