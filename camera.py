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
from typing import Callable

import cv2
import numpy as np

import settings
from image import CameraContext, LabeledImage
from mjpeg_streamer import MjpegServer, Stream


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


def start_local_capture(callback: Callable[[LabeledImage], None]):
    local_camera = cv2.VideoCapture(0)
    return start_capture(local_camera, callback)


def start_fs_capture(video_path, callback: Callable[[LabeledImage], None], target_path: str, model_path: str = None):
    fs_capture = cv2.VideoCapture(video_path)
    return start_capture(fs_capture, callback, target_path=target_path, model_path=model_path)


# TODO: fix error when using mp4 files as video source
def start_capture(video_source, on_frame: Callable[[LabeledImage], None], target_path="data.json", model_path=None):
    """Initiates the application loop

    Starts an infinite loop that accepts images from the local camera. LabeledImage instance
    is created every tick.
    """

    camera_context = CameraContext(target_path, model_path=model_path)

    # Initialize MJPEG Server
    # TODO: move to a separate function
    # stream = Stream(settings.get_mjpeg_channel_name(), size=(256, 256), quality=50, fps=30)
    # server = MjpegServer("localhost", settings.get_mjpeg_port())
    # server.add_stream(stream)
    # server.start()

    while video_source.isOpened():
        try:
            ret, frame = video_source.read()

            labeled_image = LabeledImage(preprocess_image(frame), camera_context)
            on_frame(labeled_image)

            camera_context.timer.tick_sec()  # increment timer tick

            cv2.imshow('frame', labeled_image.raw_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # stream.set_frame(labeled_image.raw_image)
        except Exception as error:
            print(error)
            camera_context.ml_labeler.save_data()
            break

    # server.stop()

    video_source.release()
    cv2.destroyAllWindows()

    return camera_context
