import unittest
import pandas as pd
from golden_cross import wykryj_golden_cross

class TestWykryjGoldenCross(unittest.TestCase):
    def setUp(self):
        self.data_golden_cross = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10', '2023-12-11'],
            'close': [100, 102, 104, 108]
        })

        self.data_no_golden_cross = pd.DataFrame({
            'date': ['2023-12-08', '2023-12-09', '2023-12-10', '2023-12-11'],
            'close': [108, 106, 104, 102]
        })

    def test_detect_golden_cross(self):
        wynik = wykryj_golden_cross(self.data_golden_cross)
        self.assertTrue(wynik, "Formacja 'golden cross' powinna zostać wykryta.")

    def test_no_golden_cross(self):
        wynik = wykryj_golden_cross(self.data_no_golden_cross)
        self.assertFalse(wynik, "Formacja 'golden cross' nie powinna zostać wykryta.")

if __name__ == '__main__':
    unittest.main()
