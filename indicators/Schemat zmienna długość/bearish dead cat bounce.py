import pandas as pd

# Funkcja wykrywająca formację "bearish dead cat bounce"
def wykryj_bearish_dead_cat_bounce(dane, liczba_swiec=12, stosunek_odbicia=0.35):
    """
    Wykrywa formację "bearish dead cat bounce" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    liczba_swiec (int): Liczba kolejnych świec w odbiciu.
    stosunek_odbicia (float): Procent odbicia w porównaniu do początkowego wzrostu.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(liczba_swiec + 2, len(dane)):
        # Sprawdzamy dużą świecę wzrostową
        poczatkowa_swieca = dane.iloc[i - liczba_swiec - 2]
        odbicie_swiec = dane.iloc[i - liczba_swiec - 1:i]
        kontynuacja_swiecy = dane.iloc[i]

        # Sprawdzenie, czy początkowa świeca jest dużą świecą wzrostową
        if poczatkowa_swieca['close'] <= poczatkowa_swieca['open']:
            continue  # Początkowa świeca nie jest wzrostowa, pomijamy

        # Obliczenie zakresu wzrostu i wysokości odbicia w dół
        zakres_wzrostu = poczatkowa_swieca['close'] - poczatkowa_swieca['open']
        maksymalny_spadek = poczatkowa_swieca['close'] - min(odbicie_swiec['low'])

        # Sprawdzenie, czy odbicie w dół jest mniejsze niż połowa wzrostu
        if maksymalny_spadek < stosunek_odbicia * zakres_wzrostu:
            continue  # Odbicie nie spełnia warunków, pomijamy

        # Sprawdzenie, czy kontynuacja wzrostu po odbiciu
        if kontynuacja_swiecy['close'] > kontynuacja_swiecy['open'] and kontynuacja_swiecy['close'] > poczatkowa_swieca['close']:
            return True  # Wykryto "bearish dead cat bounce"

    return False  # Jeśli nie wykryto formacji
