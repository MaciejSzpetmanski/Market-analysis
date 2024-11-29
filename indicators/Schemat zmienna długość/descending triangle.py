import pandas as pd
import numpy as np

# Funkcja wykrywająca malejące maksima (potencjalny descending triangle)
def wykryj_trend_spadkowy_maksima(dane, minimalna_liczba_maksimow=3):
    maksima = dane['high'].rolling(window=2).apply(lambda x: x.iloc[0] > x.iloc[1]).dropna().astype(bool)
    malejace_maksima = maksima[maksima].index
    return len(malejace_maksima) >= minimalna_liczba_maksimow

# Funkcja sprawdzająca, czy istnieje poziome wsparcie (dla descending triangle)
def wykryj_poziome_wsparcie(dane, tolerancja=0.01):
    minima = dane['low'].value_counts()
    minima = minima[minima > 1]
    if not minima.empty:
        najczestsze_minimum = minima.index[0]
        odchylenie = abs(dane['low'] - najczestsze_minimum) / najczestsze_minimum
        return (odchylenie < tolerancja).sum() > len(dane) * 0.5
    return False

# Funkcja wykrywająca formację "descending triangle"
def wykryj_descending_triangle(dane, minimalna_liczba_maksimow=3, tolerancja=0.01):
    """
    Wykrywa formację "descending triangle" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    minimalna_liczba_maksimow (int): Minimalna liczba malejących maksimów dla potwierdzenia formacji.
    tolerancja (float): Dopuszczalne odchylenie dla poziomego wsparcia.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    if wykryj_trend_spadkowy_maksima(dane, minimalna_liczba_maksimow) and wykryj_poziome_wsparcie(dane, tolerancja):
        return True
    return False
