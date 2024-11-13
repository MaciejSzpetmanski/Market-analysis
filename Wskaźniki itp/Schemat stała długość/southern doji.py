import pandas as pd

# Funkcja wykrywająca formację "southern doji"
def wykryj_southern_doji(dane, doji_procent=0.1):
    """
    Wykrywa formację "southern doji" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy typu doji.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(1, len(dane)):
        poprzednia_swieca = dane.iloc[i - 1]
        obecna_swieca = dane.iloc[i]

        # Sprawdzenie, czy obecna świeca jest typu doji (mały korpus)
        roznica_doji = abs(obecna_swieca['close'] - obecna_swieca['open'])
        zakres_doji = doji_procent * (obecna_swieca['high'] - obecna_swieca['low'])
        if roznica_doji > zakres_doji:
            continue  # Obecna świeca nie spełnia warunku doji

        # Sprawdzenie, czy poprzednia świeca jest spadkowa (trend spadkowy)
        if poprzednia_swieca['close'] >= poprzednia_swieca['open']:
            continue  # Brak spadkowej świecy przed doji, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False
