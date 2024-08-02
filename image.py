import cv2
import model

from labeler import MLBasedLabeler
from timer import Timer

class CameraContext:
    """Since a new LabeledImage class is initialized per frame, cameracontext stores shared data"""

    def __init__(self, timer: Timer):
        self.ml_labeler = MLBasedLabeler()
        self.timer = timer
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
        """returns synthesized image with time displayed on the left top corner"""

        return cv2.putText(image, f"{self.timer.ticks_passed}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))


class LabeledImage:
    """
    Image Object with helper functions
    """
    
    def __init__(self, image, camera_context: CameraContext):
        self.raw_image = image
        self.keypoints = model.parse_keypoints(image)
        self.camera_context = camera_context

    def get_activity(self):
        self.raw_image = self.camera_context.display_timer(self.raw_image)

        if "start_record" in self.camera_context.settings:
            if self.camera_context.settings["start_record"]:
                self.camera_context.ml_labeler.get_score(self.keypoints)
            else:
                self.camera_context.ml_labeler.save_data()
                del self.camera_context.settings["start_record"]

        for idx, keypoint in enumerate(self.keypoints[0][0]):
            if keypoint[2] < 0.1:
                continue
            self.raw_image = cv2.putText(self.raw_image, model.KEYPOINT_INDEX[idx], (int(keypoint[1] * 256), int(keypoint[0] * 256)), cv2.FONT_HERSHEY_PLAIN ,  1, (0, 255, 0))
            self.raw_image = cv2.circle(self.raw_image, (int(keypoint[1] * 256), int(keypoint[0] * 256)), 5, (0, 255, 0))

    def get_subactivity(self):
        pass
