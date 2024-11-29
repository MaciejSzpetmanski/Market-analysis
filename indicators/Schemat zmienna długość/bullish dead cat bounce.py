import pandas as pd

# Funkcja wykrywająca formację "bullish dead cat bounce"
def wykryj_bullish_dead_cat_bounce(dane, liczba_swiec=12, stosunek_odbicia=0.3):
    """
    Wykrywa formację "bullish dead cat bounce" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    liczba_swiec (int): Liczba kolejnych świec w odbiciu.
    stosunek_odbicia (float): Procent odbicia w porównaniu do początkowego spadku.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(liczba_swiec + 2, len(dane)):
        # Sprawdzamy dużą świecę spadkową
        poczatkowa_swieca = dane.iloc[i - liczba_swiec - 2]
        odbicie_swiec = dane.iloc[i - liczba_swiec - 1:i]
        kontynuacja_swiecy = dane.iloc[i]

        # Sprawdzenie, czy początkowa świeca jest dużą świecą spadkową
        if poczatkowa_swieca['close'] >= poczatkowa_swieca['open']:
            continue  # Początkowa świeca nie jest spadkowa, pomijamy

        # Obliczenie zakresu spadku i wysokości odbicia
        zakres_spadku = poczatkowa_swieca['open'] - poczatkowa_swieca['close']
        maksymalny_wzrost = max(odbicie_swiec['high']) - poczatkowa_swieca['close']

        # Sprawdzenie, czy odbicie jest mniejsze niż połowa spadku
        if maksymalny_wzrost < stosunek_odbicia * zakres_spadku:
            continue  # Odbicie nie spełnia warunków, pomijamy

        # Sprawdzenie, czy kontynuacja spadku po odbiciu
        if kontynuacja_swiecy['close'] < kontynuacja_swiecy['open'] and kontynuacja_swiecy['close'] < poczatkowa_swieca['close']:
            return True  # Wykryto "bullish dead cat bounce"

    return False  # Jeśli nie wykryto formacji
