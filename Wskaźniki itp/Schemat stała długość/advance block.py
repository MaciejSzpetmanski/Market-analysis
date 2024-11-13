import pandas as pd

# Funkcja wykrywająca formację "advance block" w danych cenowych
def wykryj_advance_block(dane):
    """
    Wykrywa formację "advance block" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(2, len(dane)):
        pierwsza_swieca = dane.iloc[i - 2]
        druga_swieca = dane.iloc[i - 1]
        trzecia_swieca = dane.iloc[i]

        # Sprawdzenie, czy wszystkie trzy świece są wzrostowe (close > open)
        if (pierwsza_swieca['close'] <= pierwsza_swieca['open'] or
            druga_swieca['close'] <= druga_swieca['open'] or
            trzecia_swieca['close'] <= trzecia_swieca['open']):
            continue  # Przynajmniej jedna świeca nie jest wzrostowa, pomijamy ten zestaw

        # Sprawdzenie malejącej wielkości korpusu świec (trend słabnący)
        korpus_pierwsza = abs(pierwsza_swieca['close'] - pierwsza_swieca['open'])
        korpus_druga = abs(druga_swieca['close'] - druga_swieca['open'])
        korpus_trzecia = abs(trzecia_swieca['close'] - trzecia_swieca['open'])
        
        if not (korpus_pierwsza > korpus_druga > korpus_trzecia):
            continue  # Korpusy świec nie maleją kolejno, pomijamy ten zestaw

        # Sprawdzenie, czy cienie górne rosną (presja sprzedaży wzrasta)
        cien_gorny_pierwsza = pierwsza_swieca['high'] - max(pierwsza_swieca['open'], pierwsza_swieca['close'])
        cien_gorny_druga = druga_swieca['high'] - max(druga_swieca['open'], druga_swieca['close'])
        cien_gorny_trzecia = trzecia_swieca['high'] - max(trzecia_swieca['open'], trzecia_swieca['close'])

        if not (cien_gorny_pierwsza < cien_gorny_druga < cien_gorny_trzecia):
            continue  # Cienie górne nie rosną kolejno, pomijamy ten zestaw

        # Jeśli wszystkie warunki są spełnione, formacja została wykryta
        return True

    # Jeśli nie wykryto formacji, zwróć False
    return False
