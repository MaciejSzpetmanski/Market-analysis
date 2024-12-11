import unittest
import pandas as pd
from advance_block import wykryj_advance_block


class TestWykryjAdvanceBlock(unittest.TestCase):
    def setUp(self):
        # Przykładowe dane wejściowe do testów
        self.data_advance_block = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10'],
            'open': [100, 102, 103],
            'close': [105, 104, 103.5],
            'high': [106, 105, 104],
            'low': [99, 101, 102],
            'adjusted_close': [105, 104, 103.5]
        })

        self.data_no_advance_block = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10'],
            'open': [100, 102, 101],
            'close': [105, 106, 107],
            'high': [106, 107, 108],
            'low': [99, 101, 100],
            'adjusted_close': [105, 106, 107]
        })

    def test_detect_advance_block(self):
        # Test, gdy formacja advance block jest wykrywana
        wynik = wykryj_advance_block(self.data_advance_block)
        self.assertTrue(wynik, "Formacja 'advance block' powinna zostać wykryta.")

    def test_no_advance_block(self):
        # Test, gdy formacja advance block nie jest obecna
        wynik = wykryj_advance_block(self.data_no_advance_block)
        self.assertFalse(wynik, "Formacja 'advance block' nie powinna zostać wykryta.")

if __name__ == '__main__':
    unittest.main()