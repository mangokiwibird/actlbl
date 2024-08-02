import cv2
import numpy as np

from image import CameraContext, LabeledImage
from timer import Timer

def preprocess_image(img, input_size=256):
    """
    movenet doesn't accept data with the exact same format returned by cv2, so a preprocessing step is required
    """
    processed_image = np.array(img)
    processed_image = cv2.resize(processed_image, (input_size, input_size))
    processed_image = processed_image.astype(np.uint8)

    return processed_image

def start_capture():
    """
    starts infinite loop, retrieves image from local camera
    """

    vid = cv2.VideoCapture(0)

    timer = Timer()
    camera_context = CameraContext(timer)

    while True: 
        ret, frame = vid.read()

        img = LabeledImage(preprocess_image(frame), camera_context)

        img.get_activity()

        timer.tick_sec()
        
        cv2.imshow('frame', img.raw_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    vid.release() 
    cv2.destroyAllWindows()