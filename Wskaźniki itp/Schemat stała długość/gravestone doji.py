import pandas as pd

# Funkcja wykrywająca formację "gravestone doji" w danych cenowych
def wykryj_gravestone_doji(dane, doji_procent=0.1):
    """
    Wykrywa formację "gravestone doji" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy typu doji.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(len(dane)):
        swieca = dane.iloc[i]

        # Sprawdzenie, czy otwarcie i zamknięcie są blisko siebie (świeca typu doji)
        roznica_doji = abs(swieca['close'] - swieca['open'])
        zakres_doji = doji_procent * (swieca['high'] - swieca['low'])
        if roznica_doji > zakres_doji:
            continue  # Różnica między open i close jest za duża, pomijamy

        # Sprawdzenie, czy cena minimalna jest równa otwarciu/zamknięciu (brak dolnego cienia)
        if swieca['low'] != min(swieca['open'], swieca['close']):
            continue  # Cena minimalna nie jest równa open/close, pomijamy

        # Sprawdzenie, czy górny cień jest długi
        gorny_cien = swieca['high'] - max(swieca['open'], swieca['close'])
        if gorny_cien < 2 * roznica_doji:
            continue  # Górny cień nie jest wystarczająco długi, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False
