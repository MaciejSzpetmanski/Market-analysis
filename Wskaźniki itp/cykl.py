import pandas as pd
import numpy as np

# Funkcja obliczająca średnią kroczącą
def srednia_kroczaca(dane, okres=10):
    return dane['close'].rolling(window=okres).mean()

# Funkcja obliczająca zmienność (odchylenie standardowe cen zamknięcia)
def oblicz_zmiennosc(dane, okres=10):
    return dane['close'].rolling(window=okres).std()

# Funkcja wykrywająca cykl neutralny, zbieżny lub rozbieżny
def wykryj_typ_cyklu(dane, okres=10, prog_zmiennosci=0.02):
    """
    Analizuje cykl w danych cenowych w celu określenia, czy mamy do czynienia z cyklem neutralnym, zbieżnym czy rozbieżnym.

    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    okres (int): Okres dla obliczeń średniej kroczącej i zmienności.
    prog_zmiennosci (float): Próg zmiany zmienności dla zidentyfikowania cyklu zbieżnego lub rozbieżnego.

    Zwraca:
    tuple: (bool, str) - (True, typ cyklu) jeśli wykryto cykl, (False, "") w przeciwnym wypadku.
    """
    dane = dane.copy()
    
    # Obliczenie średniej kroczącej i zmienności dla danego okresu
    dane['srednia_kroczaca'] = srednia_kroczaca(dane, okres)
    dane['zmiennosc'] = oblicz_zmiennosc(dane, okres)
    
    # Usunięcie wierszy z NaN (powstałych po obliczeniach kroczących)
    dane = dane.dropna()
    
    # Średnia zmienność w cyklu
    srednia_zmiennosc = dane['zmiennosc'].mean()
    # Zmiana zmienności na początku i końcu analizy
    zmiennosc_start = dane['zmiennosc'].iloc[0]
    zmiennosc_end = dane['zmiennosc'].iloc[-1]
    zmiana_zmiennosci = zmiennosc_end - zmiennosc_start
    
    # Określanie typu cyklu na podstawie zmienności i średniej kroczącej
    if abs(zmiana_zmiennosci) <= prog_zmiennosci * srednia_zmiennosc:
        return True, "Cykl Neutralny"
    elif zmiana_zmiennosci < -prog_zmiennosci * srednia_zmiennosc:
        return True, "Cykl Zbieżny"
    elif zmiana_zmiennosci > prog_zmiennosci * srednia_zmiennosc:
        return True, "Cykl Rozbieżny"
    else:
        return False, ""  # Brak wyraźnego cyklu
