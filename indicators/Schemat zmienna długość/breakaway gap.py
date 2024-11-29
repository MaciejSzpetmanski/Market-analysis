import pandas as pd

# Funkcja wykrywająca formację "breakaway gap"
def wykryj_breakaway_gap(dane, minimalna_luka=0.07):
    """
    Wykrywa formację "breakaway gap" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    minimalna_luka (float): Minimalna procentowa różnica dla uznania luki.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(1, len(dane)):
        poprzednia_swieca = dane.iloc[i - 1]
        obecna_swieca = dane.iloc[i]

        # Oblicz różnicę procentową między zamknięciem poprzedniej a otwarciem obecnej świecy
        luka_w_gore = (obecna_swieca['open'] - poprzednia_swieca['high']) / poprzednia_swieca['high']
        luka_w_dol = (poprzednia_swieca['low'] - obecna_swieca['open']) / poprzednia_swieca['low']

        # Sprawdzenie warunku luki wzrostowej (breakaway gap w trendzie wzrostowym)
        if luka_w_gore > minimalna_luka:
            return True  # Wykryto breakaway gap w górę

        # Sprawdzenie warunku luki spadkowej (breakaway gap w trendzie spadkowym)
        if luka_w_dol > minimalna_luka:
            return True  # Wykryto breakaway gap w dół

    return False  # Jeśli nie wykryto formacji
