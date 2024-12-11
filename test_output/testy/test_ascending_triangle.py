import unittest
import pandas as pd
from ascending_triangle import wykryj_ascending_triangle

class TestWykryjAscendingTriangle(unittest.TestCase):
    def setUp(self):
        self.data_ascending_triangle = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10'],
            'high': [105, 105, 105],
            'low': [100, 101, 102]
        })

        self.data_no_ascending_triangle = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10'],
            'high': [105, 104, 103],
            'low': [100, 99, 98]
        })

    def test_detect_ascending_triangle(self):
        wynik = wykryj_ascending_triangle(self.data_ascending_triangle)
        self.assertTrue(wynik, "Formacja 'ascending triangle' powinna zostać wykryta.")

    def test_no_ascending_triangle(self):
        wynik = wykryj_ascending_triangle(self.data_no_ascending_triangle)
        self.assertFalse(wynik, "Formacja 'ascending triangle' nie powinna zostać wykryta.")

if __name__ == '__main__':
    unittest.main()
