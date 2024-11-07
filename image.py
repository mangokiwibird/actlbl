#     Copyright (C) 2024 dolphin2410
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np

import movenet_wrapper as movenet_wrapper

from labeler.ml_labeler import MLBasedLabeler
from timer import Timer


class CameraContext:
    """Shared data for labeled image

    Since a new LabeledImage object is initialized per frame, shared data needs to be stored in a separate object.
    CameraContext stores shared data - timer, labeler etc. Also, shared variables can be accessed through the
    settings variable.

    Attributes:
        ml_labeler:
            MLBasedLabeler instance
        timer:
            Timer instance
        settings:
            Dictionary to save shared variables
    """

    def __init__(self, model_path: str = None):
        """Initializes a new CameraContext object

        Initializes a new labeler, timer and an empty dictionary named "settings", which is used to
        save extra shared data.
        """

        self.ml_labeler = MLBasedLabeler(model_path)
        self.timer = Timer()
        self.settings = {}
        
        def start_record():
            self.settings["start_record"] = True
            print("recording start!")

        def end_record():
            self.settings["start_record"] = False
            print("recording ended!!!")

        self.timer.register_action(3, start_record)
        self.timer.register_action(5, end_record)

    def display_timer(self, image):
        """Adds timer information to the upper right corner of the image in text

        Args:
            image: Image to add timer info onto

        Returns:
             Synthesized image with time displayed on the left top corner
        """

        return cv2.putText(image, f"{self.timer.ticks_passed}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))


class LabeledImage:
    """Image Object with helper functions

    The raw image could be saved in either a numpy array format or a python list format. A LabeledImage class
    contains the raw image data packaged with many helper functions that parses information from the given image.

    Attributes:
        raw_image: Raw image data in numpy array
        camera_context: CameraContext instance
        keypoints: Movenet keypoints data
    """
    
    def __init__(self, image: np.ndarray, camera_context: CameraContext):
        """Initializes a LabeledImage object

        Args:
            image: Raw image in numpy array format
            camera_context: CameraContext instance
        """

        self.raw_image = image
        self.camera_context = camera_context
        self.keypoints = movenet_wrapper.parse_keypoints(image)[0][0]

    def record_activity(self):
        """Records activity into json file format"""

        self.raw_image = self.camera_context.display_timer(self.raw_image)

        if "start_record" in self.camera_context.settings:
            if self.camera_context.settings["start_record"]:
                self.camera_context.ml_labeler.save_frame(self.keypoints)
            else:
                self.camera_context.ml_labeler.save_data()
                del self.camera_context.settings["start_record"]

        # adds keypoint marker to the raw image ( annotates image )
        for keypoint_index, keypoint_data in enumerate(self.keypoints):
            movenet_confidence = keypoint_data[2]
            if movenet_confidence < 0.1:
                continue

            keypoint_name = movenet_wrapper.KEYPOINT_INDEX_TO_NAME[keypoint_index]
            keypoint_coords = {
                "x": int(keypoint_data[1] * 256),
                "y": int(keypoint_data[0] * 256),
            }

            self.raw_image = cv2.putText(
                self.raw_image,
                keypoint_name,
                (keypoint_coords["x"], keypoint_coords["y"]),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0))

            self.raw_image = cv2.circle(
                self.raw_image,
                (keypoint_coords["x"], keypoint_coords["y"]),
                5,
                (0, 255, 0))

    def get_activity(self):
        self.camera_context.ml_labeler.save_frame(self.keypoints)
        scores = self.camera_context.ml_labeler.get_score()
        most_likely = scores["PREDICTION"]
        print(f"Predicted Class : {most_likely}")

    def get_subactivity(self):
        pass
