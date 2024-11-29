import pandas as pd

# Funkcja wykrywająca formację "exhaustion gap"
def wykryj_exhaustion_gap(dane, minimalna_luka=0.07, minimalna_seria=5):
    """
    Wykrywa formację "exhaustion gap" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    minimalna_luka (float): Minimalna procentowa różnica dla uznania luki.
    minimalna_seria (int): Minimalna liczba świec w jednym kierunku przed pojawieniem się luki.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    trend_wzrostowy = 0
    trend_spadkowy = 0

    for i in range(1, len(dane) - 1):
        poprzednia_swieca = dane.iloc[i - 1]
        obecna_swieca = dane.iloc[i]
        nastepna_swieca = dane.iloc[i + 1]

        # Zliczanie serii świec wzrostowych lub spadkowych
        if obecna_swieca['close'] > obecna_swieca['open']:
            trend_wzrostowy += 1
            trend_spadkowy = 0
        elif obecna_swieca['close'] < obecna_swieca['open']:
            trend_spadkowy += 1
            trend_wzrostowy = 0

        # Sprawdzenie luki w górę przy trendzie wzrostowym i odwrócenie na kolejnej świecy
        luka_w_gore = (obecna_swieca['open'] - poprzednia_swieca['high']) / poprzednia_swieca['high']
        if (trend_wzrostowy >= minimalna_seria and luka_w_gore > minimalna_luka and
            nastepna_swieca['close'] < nastepna_swieca['open']):  # Odwrócenie na spadkową świecę
            return True  # Wykryto "exhaustion gap" w górę

        # Sprawdzenie luki w dół przy trendzie spadkowym i odwrócenie na kolejnej świecy
        luka_w_dol = (poprzednia_swieca['low'] - obecna_swieca['open']) / poprzednia_swieca['low']
        if (trend_spadkowy >= minimalna_seria and luka_w_dol > minimalna_luka and
            nastepna_swieca['close'] > nastepna_swieca['open']):  # Odwrócenie na wzrostową świecę
            return True  # Wykryto "exhaustion gap" w dół

    return False  # Jeśli nie wykryto formacji
