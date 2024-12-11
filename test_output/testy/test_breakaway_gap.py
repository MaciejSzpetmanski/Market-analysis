import unittest
import pandas as pd
from breakaway_gap import wykryj_breakaway_gap

class TestWykryjBreakawayGap(unittest.TestCase):
    def setUp(self):
        self.data_breakaway_gap = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09'],
            'open': [110, 120],
            'close': [115, 125],
            'high': [115, 125],
            'low': [105, 115]
        })

        self.data_no_breakaway_gap = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09'],
            'open': [100, 105],
            'close': [105, 110],
            'high': [110, 115],
            'low': [95, 100]
        })

    def test_detect_breakaway_gap(self):
        wynik = wykryj_breakaway_gap(self.data_breakaway_gap)
        self.assertTrue(wynik, "Formacja 'breakaway gap' powinna zostać wykryta.")

    def test_no_breakaway_gap(self):
        wynik = wykryj_breakaway_gap(self.data_no_breakaway_gap)
        self.assertFalse(wynik, "Formacja 'breakaway gap' nie powinna zostać wykryta.")

if __name__ == '__main__':
    unittest.main()
