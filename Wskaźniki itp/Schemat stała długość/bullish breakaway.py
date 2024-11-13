import pandas as pd

# Funkcja wykrywająca formację "bullish breakaway" w danych cenowych
def wykryj_bullish_breakaway(dane):
    """
    Wykrywa formację "bullish breakaway" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(4, len(dane)):
        pierwsza_swieca = dane.iloc[i - 4]
        druga_swieca = dane.iloc[i - 3]
        trzecia_swieca = dane.iloc[i - 2]
        czwarta_swieca = dane.iloc[i - 1]
        piata_swieca = dane.iloc[i]

        # 1. Sprawdzenie pierwszej świecy (długa, spadkowa)
        if pierwsza_swieca['close'] >= pierwsza_swieca['open']:
            continue  # Pierwsza świeca nie jest spadkowa, pomijamy
        dlugosc_pierwszej = pierwsza_swieca['open'] - pierwsza_swieca['close']

        # 2. Sprawdzenie drugiej świecy (spadkowa z luką w dół)
        if druga_swieca['close'] >= druga_swieca['open'] or druga_swieca['open'] >= pierwsza_swieca['close']:
            continue  # Druga świeca nie jest spadkowa lub brak luki w dół, pomijamy

        # 3. Sprawdzenie trzeciej świecy (spadkowa z mniejszym korpusem)
        korpus_trzeciej = abs(trzecia_swieca['open'] - trzecia_swieca['close'])
        if trzecia_swieca['close'] >= trzecia_swieca['open'] or korpus_trzeciej >= dlugosc_pierwszej:
            continue  # Trzecia świeca nie jest spadkowa lub jej korpus jest zbyt duży, pomijamy

        # 4. Sprawdzenie czwartej świecy (mała, wzrostowa świeca)
        if czwarta_swieca['close'] <= czwarta_swieca['open']:
            continue  # Czwarta świeca nie jest wzrostowa, pomijamy

        # 5. Sprawdzenie piątej świecy (długa, wzrostowa świeca zamykająca się powyżej zamknięcia pierwszej świecy)
        if piata_swieca['close'] <= piata_swieca['open'] or piata_swieca['close'] <= pierwsza_swieca['close']:
            continue  # Piąta świeca nie jest wzrostowa lub zamknięcie nie jest powyżej pierwszej świecy, pomijamy

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False
