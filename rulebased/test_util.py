import unittest

import numpy as np

from movenet import KeypointGroup
from rulebased import util


class TestUtility(unittest.TestCase):
    def setUp(self):
        self.mock_keypoints_data = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [10, -10, 0],
            [10, 10, 0],
        ])
        
        # x축 방향으로 10만큼 이동
        self.mock_keypoints_data2 = np.array([
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [0, 10, 0],
            [10, 0, 0],
            [10, 20, 0],
        ])

    def test_distance(self):
        distance = util.calculate_distance(self.mock_keypoints_data, KeypointGroup.NOSE, KeypointGroup.ANKLE)

        self.assertEqual(distance, 10)

    def test_velocity(self):
        vx, vy = util.calculate_velocity(self.mock_keypoints_data, self.mock_keypoints_data2)

        self.assertEqual(vx, 10)
        self.assertEqual(vy, 0)


if __name__ == '__main__':
    unittest.main()
