import unittest
import pandas as pd
from death_cross import wykryj_death_cross

class TestDeathCross(unittest.TestCase):
    def setUp(self):
        self.data_death_cross = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10'],
            'close': [200, 195, 190],
            'open': [198, 196, 193],
            'high': [201, 197, 194],
            'low': [190, 192, 188],
            'adjusted_close': [200, 195, 190]
        })

    def test_detect_death_cross(self):
        wynik = wykryj_death_cross(self.data_death_cross)
        self.assertTrue(wynik, "Formacja 'death cross' powinna zostać wykryta.")

if __name__ == '__main__':
    unittest.main()