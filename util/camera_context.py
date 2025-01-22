import cv2
from util.actlbl_history import ActlblHistory
from util.timer import Timer


class CameraContext:
    """Shared data for labeled image

    Since a new LabeledImage object is initialized per frame, shared data needs to be stored in a separate object.
    CameraContext stores shared data.

    Attributes:
        timer:
            Timer instance
        settings:
            Dictionary to save shared variables
    """

    def __init__(self):
        """Initializes a new CameraContext object

        Initializes a new timer, a history timer and an empty dictionary named "settings", which is used to
        save extra shared data.
        """

        self.timer = Timer()
        self.settings = {}
        self.history_manager = ActlblHistory()