import pandas as pd

# Funkcja wykrywająca formację "dragonfly doji" w danych cenowych
def wykryj_dragonfly_doji(dane, doji_procent=0.1):
    """
    Wykrywa formację "dragonfly doji" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy typu doji.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(len(dane)):
        swieca = dane.iloc[i]

        # Sprawdzenie, czy otwarcie i zamknięcie są bardzo blisko siebie (świeca typu doji)
        roznica_doji = abs(swieca['close'] - swieca['open'])
        zakres_doji = doji_procent * (swieca['high'] - swieca['low'])
        if roznica_doji > zakres_doji:
            continue  # Różnica między open i close jest za duża, nie jest to doji

        # Sprawdzenie, czy cena maksymalna jest równa lub bliska otwarciu/zamknięciu (brak górnego cienia)
        if abs(swieca['high'] - max(swieca['open'], swieca['close'])) > zakres_doji:
            continue  # Cena maksymalna nie jest równa open/close, nie jest to dragonfly doji

        # Sprawdzenie, czy dolny cień jest długi
        dolny_cien = min(swieca['open'], swieca['close']) - swieca['low']
        if dolny_cien < 2 * roznica_doji:
            continue  # Dolny cień nie jest wystarczająco długi, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False
