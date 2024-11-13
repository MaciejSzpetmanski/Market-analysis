import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Funkcja wykrywająca formację odwróconą "Head and Shoulders" w danych cenowych
def wykryj_odwrocona_glowe_i_ramiona(dane, odleglosc_dolin=5, tolerancja_ramion=0.05, minimalna_glebokosc_glowy=1.05):
    """
    Wykrywa formację odwróconą "Head and Shoulders" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    odleglosc_dolin (int): Minimalna odległość między dolinami.
    tolerancja_ramion (float): Tolerancja dla podobieństwa głębokości ramion.
    minimalna_glebokosc_glowy (float): Minimalny stosunek głębokości głowy do ramion.
    
    Zwraca:
    bool: True, jeśli wykryto schemat, False w przeciwnym wypadku.
    """
    # Wykorzystujemy ceny zamknięcia do wykrywania dolin
    ceny_zamkniecia = dane['close']

    # Wykrywanie lokalnych dolin (negatywne szczyty)
    doliny, _ = find_peaks(-ceny_zamkniecia, distance=odleglosc_dolin)
    wartosci_dolin = ceny_zamkniecia.iloc[doliny]

    # Iteracja przez doliny, aby znaleźć formację odwróconą "Head and Shoulders"
    for i in range(1, len(doliny) - 1):
        lewe_ramię = wartosci_dolin.iloc[i - 1]
        glowa = wartosci_dolin.iloc[i]
        prawe_ramię = wartosci_dolin.iloc[i + 1]
        
        # Sprawdzanie warunków formacji
        ramiona_podobne = abs(lewe_ramię - prawe_ramię) <= tolerancja_ramion * lewe_ramię
        glowa_nizsza = glowa < lewe_ramię * minimalna_glebokosc_glowy and glowa < prawe_ramię * minimalna_glebokosc_glowy

        # Jeśli spełnione są warunki na formację odwróconą "Head and Shoulders", zwróć True
        if ramiona_podobne and glowa_nizsza:
            return True

    # Jeśli żaden wzór nie spełnia warunków "reverse Head and Shoulders", zwróć False
    return False

# Sprawdzenie, czy w danych występuje formacja odwróconej "Head and Shoulders"
wynik = wykryj_odwrocona_glowe_i_ramiona(dane)
print("Czy wykryto formację odwróconej 'Head and Shoulders'? :", wynik)
