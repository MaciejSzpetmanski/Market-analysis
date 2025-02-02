import pandas as pd
import numpy as np

# Funkcja wykrywająca formację "gapping up doji"
def wykryj_gapping_up_doji(dane, doji_procent=0.1):
    """
    Wykrywa formację "gapping up doji" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy typu doji.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    n = len(dane)
    gapping_up_doji = np.full(n, False)
    for i in range(1, n):
        poprzednia_swieca = dane.iloc[i - 1]
        obecna_swieca = dane.iloc[i]

        # Sprawdzenie, czy obecna świeca jest doji (bliskość open i close)
        roznica_doji = abs(obecna_swieca['close'] - obecna_swieca['open'])
        zakres_doji = doji_procent * (obecna_swieca['high'] - obecna_swieca['low'])
        if roznica_doji > zakres_doji:
            continue  # Świeca nie spełnia warunku doji

        # Sprawdzenie, czy obecna świeca otwiera się z luką w górę względem poprzedniej świecy
        if obecna_swieca['low'] <= poprzednia_swieca['high']:
            continue  # Brak luki w górę, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        gapping_up_doji[i] = True

    return gapping_up_doji
