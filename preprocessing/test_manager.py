import unittest

import numpy as np
from preprocessing.manager import Manager, history_add_padding


class TestManager(unittest.TestCase):
    def test_manager(self):
        mock_testdata = \
            [
                # The first class: x fixed, y increases
                [
                    [[10.0, 10.0, 20.0], [20.0, 10.0, 20.0], [30.0, 10.0, 20.0], [40.0, 10.0, 20.0]],  # 1fr, A
                    [[10.0, 20.0, 20.0], [20.0, 20.0, 20.0], [30.0, 20.0, 20.0], [40.0, 20.0, 20.0]],  # 2fr, A
                    [[10.0, 30.0, 20.0], [20.0, 30.0, 20.0], [30.0, 30.0, 20.0], [40.0, 30.0, 20.0]],  # 3fr, A
                    [[10.0, 40.0, 20.0], [20.0, 40.0, 20.0], [30.0, 40.0, 20.0], [40.0, 40.0, 20.0]],  # 4fr, A
                    [[10.0, 50.0, 20.0], [20.0, 50.0, 20.0], [30.0, 50.0, 20.0], [40.0, 50.0, 20.0]],  # 4fr, A
                    [[10.0, 60.0, 20.0], [20.0, 60.0, 20.0], [30.0, 60.0, 20.0], [40.0, 60.0, 20.0]],  # 4fr, A
                    [[10.0, 70.0, 20.0], [20.0, 70.0, 20.0], [30.0, 70.0, 20.0], [40.0, 70.0, 20.0]],  # 4fr, A
                    [[10.0, 80.0, 20.0], [20.0, 80.0, 20.0], [30.0, 80.0, 20.0], [40.0, 80.0, 20.0]],  # 4fr, A
                ],
                [
                    [[10.0, 40.0, 20.0], [20.0, 40.0, 20.0], [30.0, 40.0, 20.0], [40.0, 40.0, 20.0]],  # 1fr, A
                    [[10.0, 50.0, 20.0], [20.0, 50.0, 20.0], [30.0, 50.0, 20.0], [40.0, 50.0, 20.0]],  # 2fr, A
                    [[10.0, 60.0, 20.0], [20.0, 60.0, 20.0], [30.0, 60.0, 20.0], [40.0, 60.0, 20.0]],  # 3fr, A
                    [[10.0, 70.0, 20.0], [20.0, 70.0, 20.0], [30.0, 70.0, 20.0], [40.0, 70.0, 20.0]],  # 4fr, A
                    [[10.0, 80.0, 20.0], [20.0, 80.0, 20.0], [30.0, 80.0, 20.0], [40.0, 80.0, 20.0]],  # 4fr, A
                    [[10.0, 90.0, 20.0], [20.0, 90.0, 20.0], [30.0, 90.0, 20.0], [40.0, 90.0, 20.0]],  # 4fr, A
                    [[10.0, 100.0, 20.0], [20.0, 100.0, 20.0], [30.0, 100.0, 20.0], [40.0, 100.0, 20.0]],  # 4fr, A
                    [[10.0, 110.0, 20.0], [20.0, 110.0, 20.0], [30.0, 110.0, 20.0], [40.0, 110.0, 20.0]],  # 4fr, A
                ],
                [
                    [[40.0, 10.0, 20.0], [50.0, 10.0, 20.0], [60.0, 10.0, 20.0], [60.0, 10.0, 20.0]],  # 1fr, A
                    [[40.0, 20.0, 20.0], [50.0, 20.0, 20.0], [60.0, 20.0, 20.0], [60.0, 20.0, 20.0]],  # 2fr, A
                    [[40.0, 30.0, 20.0], [50.0, 30.0, 20.0], [60.0, 30.0, 20.0], [60.0, 30.0, 20.0]],  # 3fr, A
                    [[40.0, 40.0, 20.0], [50.0, 40.0, 20.0], [60.0, 40.0, 20.0], [60.0, 40.0, 20.0]],  # 3fr, A
                    [[40.0, 50.0, 20.0], [50.0, 50.0, 20.0], [60.0, 50.0, 20.0], [60.0, 50.0, 20.0]],  # 3fr, A
                    [[40.0, 60.0, 20.0], [50.0, 60.0, 20.0], [60.0, 60.0, 20.0], [60.0, 60.0, 20.0]],  # 3fr, A
                    [[40.0, 70.0, 20.0], [50.0, 70.0, 20.0], [60.0, 70.0, 20.0], [60.0, 70.0, 20.0]],  # 3fr, A
                    [[40.0, 80.0, 20.0], [50.0, 80.0, 20.0], [60.0, 80.0, 20.0], [60.0, 80.0, 20.0]],  # 3fr, A
                ],
                [
                    [[20.0, 20.0, 20.0], [30.0, 20.0, 20.0], [40.0, 20.0, 20.0], [50.0, 20.0, 20.0]],  # 1fr, A
                    [[20.0, 30.0, 20.0], [30.0, 30.0, 20.0], [40.0, 30.0, 20.0], [50.0, 30.0, 20.0]],  # 2fr, A
                    [[20.0, 40.0, 20.0], [30.0, 40.0, 20.0], [40.0, 40.0, 20.0], [50.0, 40.0, 20.0]],  # 3fr, A
                    [[20.0, 50.0, 20.0], [30.0, 50.0, 20.0], [40.0, 50.0, 20.0], [50.0, 50.0, 20.0]],  # 3fr, A
                    [[20.0, 60.0, 20.0], [30.0, 60.0, 20.0], [40.0, 60.0, 20.0], [50.0, 60.0, 20.0]],  # 3fr, A
                    [[20.0, 70.0, 20.0], [30.0, 70.0, 20.0], [40.0, 70.0, 20.0], [50.0, 70.0, 20.0]],  # 3fr, A
                    [[20.0, 80.0, 20.0], [30.0, 80.0, 20.0], [40.0, 80.0, 20.0], [50.0, 80.0, 20.0]],  # 3fr, A
                    [[20.0, 90.0, 20.0], [30.0, 90.0, 20.0], [40.0, 90.0, 20.0], [50.0, 90.0, 20.0]],  # 3fr, A
                ],
                # The second class: x fixed, y decreases
                [
                    [[10.0, 110.0, 20.0], [20.0, 110.0, 20.0], [30.0, 110.0, 20.0], [40.0, 110.0, 20.0]],  # 4fr, A
                    [[10.0, 100.0, 20.0], [20.0, 100.0, 20.0], [30.0, 100.0, 20.0], [40.0, 100.0, 20.0]],  # 4fr, A
                    [[10.0, 90.0, 20.0], [20.0, 90.0, 20.0], [30.0, 90.0, 20.0], [40.0, 90.0, 20.0]],  # 4fr, A
                    [[10.0, 80.0, 20.0], [20.0, 80.0, 20.0], [30.0, 80.0, 20.0], [40.0, 80.0, 20.0]],  # 4fr, A
                    [[10.0, 70.0, 20.0], [20.0, 70.0, 20.0], [30.0, 70.0, 20.0], [40.0, 70.0, 20.0]],  # 4fr, A
                    [[10.0, 60.0, 20.0], [20.0, 60.0, 20.0], [30.0, 60.0, 20.0], [40.0, 60.0, 20.0]],  # 3fr, A
                    [[10.0, 50.0, 20.0], [20.0, 50.0, 20.0], [30.0, 50.0, 20.0], [40.0, 50.0, 20.0]],  # 2fr, A
                    [[10.0, 40.0, 20.0], [20.0, 40.0, 20.0], [30.0, 40.0, 20.0], [40.0, 40.0, 20.0]],  # 1fr, A
                ],
                [
                    [[40.0, 80.0, 20.0], [50.0, 80.0, 20.0], [60.0, 80.0, 20.0], [60.0, 80.0, 20.0]],  # 3fr, A
                    [[40.0, 70.0, 20.0], [50.0, 70.0, 20.0], [60.0, 70.0, 20.0], [60.0, 70.0, 20.0]],  # 3fr, A
                    [[40.0, 60.0, 20.0], [50.0, 60.0, 20.0], [60.0, 60.0, 20.0], [60.0, 60.0, 20.0]],  # 3fr, A
                    [[40.0, 50.0, 20.0], [50.0, 50.0, 20.0], [60.0, 50.0, 20.0], [60.0, 50.0, 20.0]],  # 3fr, A
                    [[40.0, 40.0, 20.0], [50.0, 40.0, 20.0], [60.0, 40.0, 20.0], [60.0, 40.0, 20.0]],  # 2fr, A
                    [[40.0, 30.0, 20.0], [50.0, 30.0, 20.0], [60.0, 30.0, 20.0], [60.0, 30.0, 20.0]],  # 1fr, A
                    [[40.0, 20.0, 20.0], [50.0, 20.0, 20.0], [60.0, 20.0, 20.0], [60.0, 20.0, 20.0]],  # 1fr, A
                    [[40.0, 10.0, 20.0], [50.0, 10.0, 20.0], [60.0, 10.0, 20.0], [60.0, 10.0, 20.0]],  # 1fr, A
                ],
                [
                    [[20.0, 80.0, 20.0], [30.0, 80.0, 20.0], [40.0, 80.0, 20.0], [50.0, 80.0, 20.0]],  # 3fr, A
                    [[20.0, 70.0, 20.0], [30.0, 70.0, 20.0], [40.0, 70.0, 20.0], [50.0, 70.0, 20.0]],  # 3fr, A
                    [[20.0, 60.0, 20.0], [30.0, 60.0, 20.0], [40.0, 60.0, 20.0], [50.0, 60.0, 20.0]],  # 3fr, A
                    [[20.0, 50.0, 20.0], [30.0, 50.0, 20.0], [40.0, 50.0, 20.0], [50.0, 50.0, 20.0]],  # 3fr, A
                    [[20.0, 40.0, 20.0], [30.0, 40.0, 20.0], [40.0, 40.0, 20.0], [50.0, 40.0, 20.0]],  # 2fr, A
                    [[20.0, 30.0, 20.0], [30.0, 30.0, 20.0], [40.0, 30.0, 20.0], [50.0, 30.0, 20.0]],  # 1fr, A
                    [[20.0, 20.0, 20.0], [30.0, 20.0, 20.0], [40.0, 20.0, 20.0], [50.0, 20.0, 20.0]],  # 1fr, A
                    [[20.0, 10.0, 20.0], [30.0, 10.0, 20.0], [40.0, 10.0, 20.0], [50.0, 10.0, 20.0]],  # 1fr, A
                ],
                [
                    [[10.0, 80.0, 20.0], [20.0, 80.0, 20.0], [30.0, 80.0, 20.0], [40.0, 80.0, 20.0]],  # 4fr, A
                    [[10.0, 70.0, 20.0], [20.0, 70.0, 20.0], [30.0, 70.0, 20.0], [40.0, 70.0, 20.0]],  # 4fr, A
                    [[10.0, 60.0, 20.0], [20.0, 60.0, 20.0], [30.0, 60.0, 20.0], [40.0, 60.0, 20.0]],  # 4fr, A
                    [[10.0, 50.0, 20.0], [20.0, 50.0, 20.0], [30.0, 50.0, 20.0], [40.0, 50.0, 20.0]],  # 4fr, A
                    [[10.0, 40.0, 20.0], [20.0, 40.0, 20.0], [30.0, 40.0, 20.0], [40.0, 40.0, 20.0]],  # 4fr, A
                    [[10.0, 30.0, 20.0], [20.0, 30.0, 20.0], [30.0, 30.0, 20.0], [40.0, 30.0, 20.0]],  # 3fr, A
                    [[10.0, 20.0, 20.0], [20.0, 20.0, 20.0], [30.0, 20.0, 20.0], [40.0, 20.0, 20.0]],  # 2fr, A
                    [[10.0, 10.0, 20.0], [20.0, 10.0, 20.0], [30.0, 10.0, 20.0], [40.0, 10.0, 20.0]],  # 1fr, A
                ],
                # The third class: x decreases, y fixed
                [
                    [[110.0, 10.0, 20.0], [110.0, 20.0, 20.0], [110.0, 30.0, 20.0], [110.0, 40.0, 20.0]],  # 4fr, A
                    [[100.0, 10.0, 20.0], [100.0, 20.0, 20.0], [100.0, 30.0, 20.0], [100.0, 40.0, 20.0]],  # 4fr, A
                    [[90.0, 10.0, 20.0], [90.0, 20.0, 20.0], [90.0, 30.0, 20.0], [90.0, 40.0, 20.0]],  # 4fr, A
                    [[80.0, 10.0, 20.0], [80.0, 20.0, 20.0], [80.0, 30.0, 20.0], [80.0, 40.0, 20.0]],  # 4fr, A
                    [[70.0, 10.0, 20.0], [70.0, 20.0, 20.0], [70.0, 30.0, 20.0], [70.0, 40.0, 20.0]],  # 4fr, A
                    [[60.0, 10.0, 20.0], [60.0, 20.0, 20.0], [60.0, 30.0, 20.0], [60.0, 40.0, 20.0]],  # 3fr, A
                    [[50.0, 10.0, 20.0], [50.0, 20.0, 20.0], [50.0, 30.0, 20.0], [50.0, 40.0, 20.0]],  # 2fr, A
                    [[40.0, 10.0, 20.0], [40.0, 20.0, 20.0], [40.0, 30.0, 20.0], [40.0, 40.0, 20.0]],  # 1fr, A
                ],
                [
                    [[80.0, 40.0, 20.0], [80.0, 50.0, 20.0], [80.0, 60.0, 20.0], [80.0, 60.0, 20.0]],  # 3fr, A
                    [[70.0, 40.0, 20.0], [70.0, 50.0, 20.0], [70.0, 60.0, 20.0], [70.0, 60.0, 20.0]],  # 3fr, A
                    [[60.0, 40.0, 20.0], [60.0, 50.0, 20.0], [60.0, 60.0, 20.0], [60.0, 60.0, 20.0]],  # 3fr, A
                    [[50.0, 40.0, 20.0], [50.0, 50.0, 20.0], [50.0, 60.0, 20.0], [50.0, 60.0, 20.0]],  # 3fr, A
                    [[40.0, 40.0, 20.0], [40.0, 50.0, 20.0], [40.0, 60.0, 20.0], [40.0, 60.0, 20.0]],  # 2fr, A
                    [[30.0, 40.0, 20.0], [30.0, 50.0, 20.0], [30.0, 60.0, 20.0], [30.0, 60.0, 20.0]],  # 1fr, A
                    [[20.0, 40.0, 20.0], [20.0, 50.0, 20.0], [20.0, 60.0, 20.0], [20.0, 60.0, 20.0]],  # 1fr, A
                    [[10.0, 40.0, 20.0], [10.0, 50.0, 20.0], [10.0, 60.0, 20.0], [10.0, 60.0, 20.0]],  # 1fr, A
                ],
                [
                    [[80.0, 20.0, 20.0], [80.0, 30.0, 20.0], [80.0, 40.0, 20.0], [80.0, 50.0, 20.0]],  # 3fr, A
                    [[70.0, 20.0, 20.0], [70.0, 30.0, 20.0], [70.0, 40.0, 20.0], [70.0, 50.0, 20.0]],  # 3fr, A
                    [[60.0, 20.0, 20.0], [60.0, 30.0, 20.0], [60.0, 40.0, 20.0], [60.0, 50.0, 20.0]],  # 3fr, A
                    [[50.0, 20.0, 20.0], [50.0, 30.0, 20.0], [50.0, 40.0, 20.0], [50.0, 50.0, 20.0]],  # 3fr, A
                    [[40.0, 20.0, 20.0], [40.0, 30.0, 20.0], [40.0, 40.0, 20.0], [40.0, 50.0, 20.0]],  # 2fr, A
                    [[30.0, 20.0, 20.0], [30.0, 30.0, 20.0], [30.0, 40.0, 20.0], [30.0, 50.0, 20.0]],  # 1fr, A
                    [[20.0, 20.0, 20.0], [20.0, 30.0, 20.0], [20.0, 40.0, 20.0], [20.0, 50.0, 20.0]],  # 1fr, A
                    [[10.0, 20.0, 20.0], [10.0, 30.0, 20.0], [10.0, 40.0, 20.0], [10.0, 50.0, 20.0]],  # 1fr, A
                ],
                [
                    [[80.0, 10.0, 20.0], [80.0, 20.0, 20.0], [80.0, 30.0, 20.0], [80.0, 40.0, 20.0]],  # 4fr, A
                    [[70.0, 10.0, 20.0], [70.0, 20.0, 20.0], [70.0, 30.0, 20.0], [70.0, 40.0, 20.0]],  # 4fr, A
                    [[60.0, 10.0, 20.0], [60.0, 20.0, 20.0], [60.0, 30.0, 20.0], [60.0, 40.0, 20.0]],  # 4fr, A
                    [[50.0, 10.0, 20.0], [50.0, 20.0, 20.0], [50.0, 30.0, 20.0], [50.0, 40.0, 20.0]],  # 4fr, A
                    [[40.0, 10.0, 20.0], [40.0, 20.0, 20.0], [40.0, 30.0, 20.0], [40.0, 40.0, 20.0]],  # 4fr, A
                    [[30.0, 10.0, 20.0], [30.0, 20.0, 20.0], [30.0, 30.0, 20.0], [30.0, 40.0, 20.0]],  # 3fr, A
                    [[20.0, 10.0, 20.0], [20.0, 20.0, 20.0], [20.0, 30.0, 20.0], [20.0, 40.0, 20.0]],  # 2fr, A
                    [[10.0, 10.0, 20.0], [10.0, 20.0, 20.0], [10.0, 30.0, 20.0], [10.0, 40.0, 20.0]],  # 1fr, A
                ],
                # The fourth class: x increases, y fixed
                [
                    [[10.0, 10.0, 20.0], [10.0, 20.0, 20.0], [10.0, 30.0, 20.0], [10.0, 40.0, 20.0]],  # 1fr, A
                    [[20.0, 10.0, 20.0], [20.0, 20.0, 20.0], [20.0, 30.0, 20.0], [20.0, 40.0, 20.0]],  # 2fr, A
                    [[30.0, 10.0, 20.0], [30.0, 20.0, 20.0], [30.0, 30.0, 20.0], [30.0, 40.0, 20.0]],  # 3fr, A
                    [[40.0, 10.0, 20.0], [40.0, 20.0, 20.0], [40.0, 30.0, 20.0], [40.0, 40.0, 20.0]],  # 4fr, A
                    [[50.0, 10.0, 20.0], [50.0, 20.0, 20.0], [50.0, 30.0, 20.0], [50.0, 40.0, 20.0]],  # 4fr, A
                    [[60.0, 10.0, 20.0], [60.0, 20.0, 20.0], [60.0, 30.0, 20.0], [60.0, 40.0, 20.0]],  # 4fr, A
                    [[70.0, 10.0, 20.0], [70.0, 20.0, 20.0], [70.0, 30.0, 20.0], [70.0, 40.0, 20.0]],  # 4fr, A
                    [[80.0, 10.0, 20.0], [80.0, 20.0, 20.0], [80.0, 30.0, 20.0], [80.0, 40.0, 20.0]],  # 4fr, A
                ],
                [
                    [[40.0, 10.0, 20.0], [40.0, 20.0, 20.0], [40.0, 30.0, 20.0], [40.0, 40.0, 20.0]],  # 1fr, A
                    [[50.0, 10.0, 20.0], [50.0, 20.0, 20.0], [50.0, 30.0, 20.0], [50.0, 40.0, 20.0]],  # 2fr, A
                    [[60.0, 10.0, 20.0], [60.0, 20.0, 20.0], [60.0, 30.0, 20.0], [60.0, 40.0, 20.0]],  # 3fr, A
                    [[70.0, 10.0, 20.0], [70.0, 20.0, 20.0], [70.0, 30.0, 20.0], [70.0, 40.0, 20.0]],  # 4fr, A
                    [[80.0, 10.0, 20.0], [80.0, 20.0, 20.0], [80.0, 30.0, 20.0], [80.0, 40.0, 20.0]],  # 4fr, A
                    [[90.0, 10.0, 20.0], [90.0, 20.0, 20.0], [90.0, 30.0, 20.0], [90.0, 40.0, 20.0]],  # 4fr, A
                    [[100.0, 10.0, 20.0], [100.0, 20.0, 20.0], [100.0, 30.0, 20.0], [100.0, 40.0, 20.0]],  # 4fr, A
                    [[110.0, 10.0, 20.0], [110.0, 20.0, 20.0], [110.0, 30.0, 20.0], [110.0, 40.0, 20.0]],  # 4fr, A
                ],
                [
                    [[10.0, 40.0, 20.0], [10.0, 50.0, 20.0], [10.0, 60.0, 20.0], [10.0, 60.0, 20.0]],  # 1fr, A
                    [[20.0, 40.0, 20.0], [20.0, 50.0, 20.0], [20.0, 60.0, 20.0], [20.0, 60.0, 20.0]],  # 2fr, A
                    [[30.0, 40.0, 20.0], [30.0, 50.0, 20.0], [30.0, 60.0, 20.0], [30.0, 60.0, 20.0]],  # 3fr, A
                    [[40.0, 40.0, 20.0], [40.0, 50.0, 20.0], [40.0, 60.0, 20.0], [40.0, 60.0, 20.0]],  # 3fr, A
                    [[50.0, 40.0, 20.0], [50.0, 50.0, 20.0], [50.0, 60.0, 20.0], [50.0, 60.0, 20.0]],  # 3fr, A
                    [[60.0, 40.0, 20.0], [60.0, 50.0, 20.0], [60.0, 60.0, 20.0], [60.0, 60.0, 20.0]],  # 3fr, A
                    [[70.0, 40.0, 20.0], [70.0, 50.0, 20.0], [70.0, 60.0, 20.0], [70.0, 60.0, 20.0]],  # 3fr, A
                    [[80.0, 40.0, 20.0], [80.0, 50.0, 20.0], [80.0, 60.0, 20.0], [80.0, 60.0, 20.0]],  # 3fr, A
                ],
                [
                    [[20.0, 20.0, 20.0], [20.0, 30.0, 20.0], [20.0, 40.0, 20.0], [20.0, 50.0, 20.0]],  # 1fr, A
                    [[30.0, 20.0, 20.0], [30.0, 30.0, 20.0], [30.0, 40.0, 20.0], [30.0, 50.0, 20.0]],  # 2fr, A
                    [[40.0, 20.0, 20.0], [40.0, 30.0, 20.0], [40.0, 40.0, 20.0], [40.0, 50.0, 20.0]],  # 3fr, A
                    [[50.0, 20.0, 20.0], [50.0, 30.0, 20.0], [50.0, 40.0, 20.0], [50.0, 50.0, 20.0]],  # 3fr, A
                    [[60.0, 20.0, 20.0], [60.0, 30.0, 20.0], [60.0, 40.0, 20.0], [60.0, 50.0, 20.0]],  # 3fr, A
                    [[70.0, 20.0, 20.0], [70.0, 30.0, 20.0], [70.0, 40.0, 20.0], [70.0, 50.0, 20.0]],  # 3fr, A
                ]
            ]

        mock_answers = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

        manager = Manager(np.array(history_add_padding(mock_testdata)), np.array(mock_answers))
        accuracy = manager.generate_model()

        print(f"Accuracy: {accuracy}")

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
