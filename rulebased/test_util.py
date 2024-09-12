import unittest

import numpy as np

import movenet
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

    def test_center_location(self):
        center_coords = util.get_center_coords(movenet.objectify_keypoints(self.mock_keypoints_data), KeypointGroup.ANKLE)
        self.assertEqual(center_coords.x, 0)
        self.assertEqual(center_coords.y, 10)

    def test_distance(self):
        distance = util.calculate_distance(self.mock_keypoints_data, KeypointGroup.NOSE, KeypointGroup.ANKLE)

        self.assertEqual(distance, 10)

    def test_velocity(self):
        vx, vy = util.calculate_velocity(self.mock_keypoints_data, self.mock_keypoints_data2)

        self.assertEqual(vx, 10)
        self.assertEqual(vy, 0)


if __name__ == '__main__':
    unittest.main()
