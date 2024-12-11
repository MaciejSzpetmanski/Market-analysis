import unittest
import pandas as pd
from double_bottom import wykryj_podwojne_dno

class TestDoubleBottom(unittest.TestCase):
    def setUp(self):
        self.data_double_bottom = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10', '2023-12-11', '2023-12-12'],
            'low': [100, 95, 100, 96, 100],
            'high': [110, 115, 112, 114, 110],
            'open': [105, 100, 105, 102, 105],
            'close': [106, 98, 105, 103, 104],
            'adjusted_close': [106, 98, 105, 103, 104]
        })

    def test_detect_double_bottom(self):
        wynik = wykryj_podwojne_dno(self.data_double_bottom)
        self.assertTrue(wynik, "Formacja 'double bottom' powinna zostać wykryta.")

if __name__ == '__main__':
    unittest.main()