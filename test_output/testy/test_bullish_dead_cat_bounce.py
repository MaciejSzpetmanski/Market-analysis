import unittest
import pandas as pd
from bullish_dead_cat_bounce import wykryj_bullish_dead_cat_bounce

class TestBullishDeadCatBounce(unittest.TestCase):
    def setUp(self):
        self.data_bullish_dcb = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10', '2023-12-11'],
            'open': [100, 98, 97, 95],
            'close': [98, 97, 96, 94],
            'high': [99, 98, 97, 96],
            'low': [95, 94, 93, 92],
            'adjusted_close': [98, 97, 96, 94]
        })

    def test_detect_bullish_dead_cat_bounce(self):
        wynik = wykryj_bullish_dead_cat_bounce(self.data_bullish_dcb)
        self.assertFalse(wynik, "Formacja 'bullish dead cat bounce' nie powinna zostać wykryta w tych danych.")

if __name__ == '__main__':
    unittest.main()