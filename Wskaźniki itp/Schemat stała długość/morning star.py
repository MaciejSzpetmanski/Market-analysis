import pandas as pd

# Funkcja wykrywająca formację "morning star"
def wykryj_morning_star(dane, doji_procent=0.1):
    """
    Wykrywa formację "morning star" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    doji_procent (float): Procentowy zakres dla różnicy między open i close w świecy o małym korpusie.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(2, len(dane)):
        pierwsza_swieca = dane.iloc[i - 2]
        druga_swieca = dane.iloc[i - 1]
        trzecia_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (spadkowa)
        if pierwsza_swieca['close'] >= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest spadkowa, pomijamy

        # 2. Sprawdzenie drugiej świecy (mały korpus, potencjalne doji)
        roznica_doja = abs(druga_swieca['close'] - druga_swieca['open'])
        zakres_doja = doji_procent * (druga_swieca['high'] - druga_swieca['low'])
        if roznica_doja > zakres_doja:
            continue  # Druga świeca nie jest doji lub ma mały korpus, pomijamy

        # 3. Sprawdzenie trzeciej świecy (wzrostowa, zamykająca się powyżej połowy korpusu pierwszej świecy)
        polowa_korpusu_pierwszej = pierwsza_swieca['open'] + 0.5 * (pierwsza_swieca['close'] - pierwsza_swieca['open'])
        if trzecia_swieca['close'] <= trzecia_swieca['open'] or trzecia_swieca['close'] <= polowa_korpusu_pierwszej:
            continue  # Trzecia świeca nie jest wzrostowa lub zamyka się poniżej środka pierwszej świecy, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False
