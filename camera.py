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

from image import CameraContext, LabeledImage


# TODO: is this step required?
def preprocess_image(
        image: np.ndarray,
        input_size: int = 256,
):
    """Preprocesses image returned by cv2

    Args:
        image: Image to process
        input_size: Input size in pixels

    Returns:
        The processed image

    """

    processed_image = np.array(image)
    processed_image = cv2.resize(processed_image, (input_size, input_size))
    processed_image = processed_image.astype(np.uint8)

    return processed_image


def start_capture():
    """Initiates the application loop

    Starts an infinite loop that accepts images from the local camera. LabeledImage instance
    is created every tick.
    """

    local_camera = cv2.VideoCapture(0)
    camera_context = CameraContext()

    while True:
        ret, frame = local_camera.read()

        labeled_image = LabeledImage(preprocess_image(frame), camera_context)
        labeled_image.get_activity()

        camera_context.timer.tick_sec()  # increment timer tick

        cv2.imshow('frame', labeled_image.raw_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    local_camera.release()
    cv2.destroyAllWindows()
