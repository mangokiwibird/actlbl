import unittest

import numpy as np

import masking


class TestMasking(unittest.TestCase):
    def test_padding(self):
        mock_testdata = \
            [
                [
                    [[10.0, 10.0, 0.2], [10.0, 20.0, 0.2], [20.0, 10.0, 0.2], [20.0, 20.0, 0.2]],  # 1fr, A
                    [[10.0, 10.0, 0.2], [10.0, 20.0, 0.2], [20.0, 10.0, 0.2], [20.0, 20.0, 0.2]],  # 2fr, A
                    [[10.0, 10.0, 0.2], [10.0, 20.0, 0.2], [20.0, 10.0, 0.2], [20.0, 20.0, 0.2]],  # 3fr, A
                    [[10.0, 10.0, 0.2], [10.0, 20.0, 0.2], [20.0, 10.0, 0.2], [20.0, 20.0, 0.2]],  # 4fr, A
                ],
                [
                    [[10.0, 10.0, 0.2], [10.0, 20.0, 0.2], [20.0, 10.0, 0.2], [20.0, 20.0, 0.2]],  # 1fr, B
                    [[10.0, 10.0, 0.2], [10.0, 20.0, 0.2], [20.0, 10.0, 0.2], [20.0, 20.0, 0.2]],  # 2fr, B
                ]
            ]

        padded_data = masking.history_add_padding(mock_testdata)

        self.assertEqual(np.array(padded_data).shape, (2, 4, 4, 3))  # add assertion here

        masking.create_mask_layer(padded_data)


if __name__ == '__main__':
    unittest.main()
