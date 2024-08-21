from camera import start_local_capture
from image import LabeledImage


def train_from_local():

    def callback(img: LabeledImage):
        print(img.get_activity())

    start_local_capture(callback)
