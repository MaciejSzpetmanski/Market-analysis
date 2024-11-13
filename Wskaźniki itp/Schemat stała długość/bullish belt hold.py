import pandas as pd

# Funkcja wykrywająca formację "bullish belt hold" w danych cenowych
def wykryj_bullish_belt_hold(dane):
    """
    Wykrywa formację "bullish belt hold" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(len(dane)):
        swieca = dane.iloc[i]

        # Sprawdzenie, czy świeca jest wzrostowa (close > open)
        if swieca['close'] <= swieca['open']:
            continue  # Świeca nie jest wzrostowa, pomijamy

        # Sprawdzenie, czy świeca otwiera się na najniższym poziomie (brak dolnego cienia)
        if swieca['open'] != swieca['low']:
            continue  # Świeca nie otwiera się na minimum, pomijamy

        # Sprawdzenie, czy świeca jest długa, czyli zamyka się znacznie wyżej
        dlugosc_swiecy = swieca['close'] - swieca['open']
        srednia_zmiennosc = (swieca['high'] - swieca['low']) * 0.5  # Średnia zmienność świecowa
        if dlugosc_swiecy < srednia_zmiennosc:
            continue  # Świeca nie jest wystarczająco długa, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False
