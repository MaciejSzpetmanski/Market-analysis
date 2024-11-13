import pandas as pd

# Funkcja wykrywająca formację "collapsing doji star"
def wykryj_collapsing_doji_star(dane, doji_procent=0.1):
    """
    Wykrywa formację "collapsing doji star" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy typu doji.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(2, len(dane)):
        pierwsza_swieca = dane.iloc[i - 2]
        druga_swieca = dane.iloc[i - 1]
        trzecia_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (długa, wzrostowa)
        if pierwsza_swieca['close'] <= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest wzrostowa, pomijamy

        # 2. Sprawdzenie drugiej świecy (doji z małym korpusem)
        roznica_doji = abs(druga_swieca['close'] - druga_swieca['open'])
        zakres_doji = doji_procent * (druga_swieca['high'] - druga_swieca['low'])
        if roznica_doji > zakres_doji:
            continue  # Druga świeca nie jest doji, pomijamy

        # 3. Sprawdzenie trzeciej świecy (spadkowa świeca, zamknięcie poniżej zamknięcia pierwszej świecy)
        if trzecia_swieca['close'] >= trzecia_swieca['open'] or trzecia_swieca['close'] >= pierwsza_swieca['close']:
            continue  # Trzecia świeca nie jest spadkowa lub nie zamyka się poniżej zamknięcia pierwszej świecy, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False

