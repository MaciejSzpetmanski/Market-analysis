import unittest
import pandas as pd
from bearish_dead_cat_bounce import wykryj_bearish_dead_cat_bounce

class TestWykryjBearishDeadCatBounce(unittest.TestCase):
    def setUp(self):
        self.data_bearish_dead_cat = pd.DataFrame({
            'date': ['2023-12-01', '2023-12-02', '2023-12-03'],
            'open': [100, 110, 90],
            'close': [110, 90, 85],
            'high': [115, 110, 95],
            'low': [95, 85, 80]
        })

        self.data_no_bearish_dead_cat = pd.DataFrame({
            'date': ['2023-12-01', '2023-12-02', '2023-12-03'],
            'open': [90, 85, 80],
            'close': [100, 95, 90],
            'high': [105, 100, 95],
            'low': [85, 80, 75]
        })

    def test_detect_bearish_dead_cat_bounce(self):
        wynik = wykryj_bearish_dead_cat_bounce(self.data_bearish_dead_cat)
        self.assertTrue(wynik, "Formacja 'bearish dead cat bounce' powinna zostać wykryta.")

    def test_no_bearish_dead_cat_bounce(self):
        wynik = wykryj_bearish_dead_cat_bounce(self.data_no_bearish_dead_cat)
        self.assertFalse(wynik, "Formacja 'bearish dead cat bounce' nie powinna zostać wykryta.")

if __name__ == '__main__':
    unittest.main()
