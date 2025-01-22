import cv2

from util.camera_context import CameraContext
from util.actlbl_image import ActlblImage
from util.movenet_wrapper import resize_for_movenet


class ActlblApplication:
    def __init__(self, video_source):
        self.video_source = video_source
        self.camera_context = CameraContext()

    def start_application(self, application_callback = None, exit_callback = None):  # todo: maybe non-blocking code?
        """
        application_callback: ActlblImage -> None
        exit_callback: Actlbl Application -> None
        """
        
        self.is_valid = True
        
        video_capture = cv2.VideoCapture(self.video_source)


        while video_capture.isOpened():
            try:
                ret, frame = video_capture.read()

                labeled_image = ActlblImage(resize_for_movenet(frame), self.camera_context)
                
                if application_callback is not None:
                    application_callback(labeled_image)

                self.camera_context.timer.tick_sec()  # increment timer tick

                cv2.imshow('frame', labeled_image.raw_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as error:
                print(error)
                break
        
        if exit_callback is not None:
            exit_callback(self)

        video_capture.release()
        cv2.destroyAllWindows()
    