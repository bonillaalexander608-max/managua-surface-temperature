import unittest

import pandas as pd

from evaluar_temporal import historical_features


class HistoricalFeaturesTest(unittest.TestCase):
    def test_history_excludes_current_and_future_values(self):
        train = pd.DataFrame({"id_celda": ["a", "b", "a"], "target": [10., 99., 20.]})
        test = pd.DataFrame({"id_celda": ["a", "b", "c"], "target": [500., 500., 500.]})
        fitted, future = historical_features(train, test, "target", "history")
        self.assertTrue(fitted.history.iloc[:2].isna().all())
        self.assertEqual(fitted.history.iloc[2], 10.)
        self.assertEqual(future.history.iloc[:2].tolist(), [15., 99.])
        self.assertTrue(pd.isna(future.history.iloc[2]))


if __name__ == "__main__":
    unittest.main()
