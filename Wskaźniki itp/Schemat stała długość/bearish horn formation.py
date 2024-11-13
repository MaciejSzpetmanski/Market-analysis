import pandas as pd 

# Funkcja wykrywająca formację "bearish horn formation"
def wykryj_bearish_horn_formation(dane, minimalny_cien=1.5):
    """
    Wykrywa formację "bearish horn formation" w dostarczonym DataFrame.
    
    Parametry:
    dane (pd.DataFrame): DataFrame z kolumnami 'date', 'open', 'close', 'high', 'low', 'adjusted_close'.
    minimalny_cien (float): Minimalny stosunek górnego cienia do korpusu świecy.
    
    Zwraca:
    bool: True, jeśli wykryto formację, False w przeciwnym wypadku.
    """
    for i in range(2, len(dane)):
        pierwsza_swieca = dane.iloc[i - 2]
        srodkowa_swieca = dane.iloc[i - 1]
        trzecia_swieca = dane.iloc[i]

        # Sprawdzenie górnych cieni dla pierwszej i trzeciej świecy
        korpus_pierwszej = abs(pierwsza_swieca['close'] - pierwsza_swieca['open'])
        gorny_cien_pierwszej = pierwsza_swieca['high'] - pierwsza_swieca['close'] if pierwsza_swieca['close'] > pierwsza_swieca['open'] else pierwsza_swieca['high'] - pierwsza_swieca['open']

        korpus_trzeciej = abs(trzecia_swieca['close'] - trzecia_swieca['open'])
        gorny_cien_trzeciej = trzecia_swieca['high'] - trzecia_swieca['close'] if trzecia_swieca['close'] > trzecia_swieca['open'] else trzecia_swieca['high'] - trzecia_swieca['open']

        # Sprawdzenie, czy górne cienie są wystarczająco długie
        if gorny_cien_pierwszej < minimalny_cien * korpus_pierwszej or gorny_cien_trzeciej < minimalny_cien * korpus_trzeciej:
            continue  # Jeśli górne cienie nie są wystarczająco długie, pomijamy

        # Sprawdzenie, czy środkowa świeca ma mniejszy korpus
        korpus_srodkowej = abs(srodkowa_swieca['close'] - srodkowa_swieca['open'])
        if korpus_srodkowej >= korpus_pierwszej or korpus_srodkowej >= korpus_trzeciej:
            continue  # Środkowa świeca nie ma mniejszego korpusu

        return True  # Wykryto "bearish horn formation"

    return False  # Jeśli nie wykryto formacji
