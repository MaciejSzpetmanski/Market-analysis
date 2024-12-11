import unittest
import pandas as pd
from descending_triangle import wykryj_descending_triangle

class TestDescendingTriangle(unittest.TestCase):
    def setUp(self):
        self.data_desc_triangle = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10'],
            'open': [105, 104, 103],
            'close': [102, 101, 100],
            'high': [106, 105, 104],
            'low': [101, 100, 99],
            'adjusted_close': [102, 101, 100]
        })

    def test_detect_descending_triangle(self):
        wynik = wykryj_descending_triangle(self.data_desc_triangle)
        self.assertTrue(wynik, "Formacja 'descending triangle' powinna zostać wykryta.")

if __name__ == '__main__':
    unittest.main()