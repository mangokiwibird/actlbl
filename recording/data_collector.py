import cv2
from core.application import ActlblApplication
from util import movenet_wrapper
from util.actlbl_history import ActlblHistory, ActlblHistoryFrame
from util.actlbl_image import ActlblImage


def application_callback(image: ActlblImage):
    camera_context = image.camera_context

    image.raw_image = camera_context.timer.display_timer(image.raw_image)

    if "start_record" in camera_context.settings:
        if camera_context.settings["start_record"]:
            frame = ActlblHistoryFrame(image.keypoints)
            camera_context.history_manager.append_frame(frame)
        else:
            camera_context.history_manager.save_history()
            del camera_context.settings["start_record"]

    # adds keypoint marker to the raw image ( annotates image )
    for keypoint_index, keypoint_data in enumerate(image.keypoints):
        movenet_confidence = keypoint_data[2]
        if movenet_confidence < 0.1:
            continue

        keypoint_name = movenet_wrapper.KEYPOINT_INDEX_TO_NAME[keypoint_index]
        keypoint_coords = {
            "x": int(keypoint_data[1] * 256),
            "y": int(keypoint_data[0] * 256),
        }

        image.raw_image = cv2.putText(
            image.raw_image,
            keypoint_name,
            (keypoint_coords["x"], keypoint_coords["y"]),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 0))

        image.raw_image = cv2.circle(
            image.raw_image,
            (keypoint_coords["x"], keypoint_coords["y"]),
            5,
            (0, 255, 0))
        
def exit_callback(application: ActlblApplication):
    camera_context = application.camera_context

    if "start_record" in camera_context.settings and camera_context.settings["start_record"]:
        camera_context.history_manager.save_history()
        del camera_context.settings["start_record"]

class LiveDataCollector():
    def __init__(self):
        self.application = ActlblApplication(0)

    def start_collector(self):
        camera_context = self.application.camera_context

        def start_record():
            camera_context.settings["start_record"] = True
            print("recording start!")

        def end_record():
            camera_context.settings["start_record"] = False
            print("recording ended!!!")

        camera_context.timer.register_action(3, start_record)
        camera_context.timer.register_action(5, end_record)

        self.application.start_application(application_callback, exit_callback)

class VideoDataCollector():
    def __init__(self, video_path):
        self.video_path = video_path
        self.application = ActlblApplication(video_path)

    def start_collector(self):
        self.application.camera_context.settings["start_record"] = True
        self.application.start_application(application_callback, exit_callback)
