import pandas as pd

# Funkcja wykrywająca formację "bearish abandoned baby" w danych cenowych
def wykryj_bearish_abandoned_baby(dane, doji_procent=0.15):
    """
    Wykrywa formację "bearish abandoned baby" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy typu doji.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(1, len(dane) - 1):
        pierwsza_swieca = dane.iloc[i - 1]
        druga_swieca = dane.iloc[i]
        trzecia_swieca = dane.iloc[i + 1]

        # Sprawdzenie pierwszej świecy (silna wzrostowa świeca)
        if pierwsza_swieca['close'] <= pierwsza_swieca['open']:
            continue  # Świeca nie jest wzrostowa, pomijamy ten zestaw

        # Sprawdzenie drugiej świecy (doji z luką w górę)
        roznica_doji = abs(druga_swieca['close'] - druga_swieca['open'])
        zakres_doji = doji_procent * (druga_swieca['high'] - druga_swieca['low'])
        if roznica_doji > zakres_doji:
            continue  # Druga świeca nie jest doji, pomijamy ten zestaw
        if druga_swieca['low'] <= pierwsza_swieca['high']:
            continue  # Brak luki w górę, pomijamy ten zestaw

        # Sprawdzenie trzeciej świecy (silna spadkowa świeca z luką w dół)
        if trzecia_swieca['close'] >= trzecia_swieca['open']:
            continue  # Świeca nie jest spadkowa, pomijamy ten zestaw
        if trzecia_swieca['high'] >= druga_swieca['low']:
            continue  # Brak luki w dół, pomijamy ten zestaw

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False
