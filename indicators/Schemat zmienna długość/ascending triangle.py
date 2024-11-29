import pandas as pd
import numpy as np

# Funkcja wykrywająca rosnące minima (potencjalny ascending triangle)
def wykryj_trend_wzrostowy_minima(dane, minimalna_liczba_minimow=3):
    minima = dane['low'].rolling(window=2).apply(lambda x: x.iloc[0] < x.iloc[1]).dropna().astype(bool)
    rosnace_minima = minima[minima].index
    return len(rosnace_minima) >= minimalna_liczba_minimow

# Funkcja sprawdzająca, czy istnieje poziomy opór (dla ascending triangle)
def wykryj_poziomy_opor(dane, tolerancja=0.01):
    maksima = dane['high'].value_counts()
    maksima = maksima[maksima > 1]
    if not maksima.empty:
        najczestsze_maksimum = maksima.index[0]
        odchylenie = abs(dane['high'] - najczestsze_maksimum) / najczestsze_maksimum
        return (odchylenie < tolerancja).sum() > len(dane) * 0.5
    return False

# Funkcja wykrywająca formację "ascending triangle"
def wykryj_ascending_triangle(dane, minimalna_liczba_minimow=3, tolerancja=0.01):
    """
    Wykrywa formację "ascending triangle" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    minimalna_liczba_minimow (int): Minimalna liczba rosnących minimów dla potwierdzenia formacji.
    tolerancja (float): Dopuszczalne odchylenie dla poziomego oporu.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    if wykryj_trend_wzrostowy_minima(dane, minimalna_liczba_minimow) and wykryj_poziomy_opor(dane, tolerancja):
        return True
    return False
