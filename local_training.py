from camera import start_local_capture
from image import LabeledImage


def record_from_local():

    def callback(img: LabeledImage):
        img.record_activity()

        if "counter" in img.camera_context.settings:
            img.camera_context.settings["counter"] += 1
        else:
            img.camera_context.settings["counter"] = 1

    ctx = start_local_capture(callback)

    print(f"Counter: {ctx.camera_context.settings["counter"]}")
