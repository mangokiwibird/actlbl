import cv2
import numpy as np

from util import movenet_wrapper
from util.camera_context import CameraContext


class ActlblImage:
    """
    Wrapper class for images

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
