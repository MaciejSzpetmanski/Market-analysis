import pandas as pd

# Funkcja wykrywająca formację "runaway gap"
def wykryj_runaway_gap(dane, minimalna_luka=0.07, minimalna_seria=5):
    """
    Wykrywa formację "runaway gap" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    minimalna_luka (float): Minimalna procentowa różnica dla uznania luki.
    minimalna_seria (int): Minimalna liczba świec w jednym kierunku przed pojawieniem się luki.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    trend_wzrostowy = 0
    trend_spadkowy = 0

    for i in range(1, len(dane)):
        poprzednia_swieca = dane.iloc[i - 1]
        obecna_swieca = dane.iloc[i]

        # Zliczanie serii świec wzrostowych lub spadkowych
        if obecna_swieca['close'] > obecna_swieca['open']:
            trend_wzrostowy += 1
            trend_spadkowy = 0
        elif obecna_swieca['close'] < obecna_swieca['open']:
            trend_spadkowy += 1
            trend_wzrostowy = 0

        # Sprawdzenie warunku luki w górę podczas trendu wzrostowego
        luka_w_gore = (obecna_swieca['open'] - poprzednia_swieca['high']) / poprzednia_swieca['high']
        if trend_wzrostowy >= minimalna_seria and luka_w_gore > minimalna_luka:
            return True  # Wykryto runaway gap w górę

        # Sprawdzenie warunku luki w dół podczas trendu spadkowego
        luka_w_dol = (poprzednia_swieca['low'] - obecna_swieca['open']) / poprzednia_swieca['low']
        if trend_spadkowy >= minimalna_seria and luka_w_dol > minimalna_luka:
            return True  # Wykryto runaway gap w dół

    return False  # Jeśli nie wykryto formacji
