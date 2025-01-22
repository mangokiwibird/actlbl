class ActlblHistoryFrame():
    """A wrapper class for movenet keypoints"""

    def __init__(self, keypoints):
        self.keypoints = keypoints

class ActlblHistory():
    """This class stores a time series of movenet keypoints"""
    
    def __init__(self):
        self.frames = []

    def append_frame(self, frame: ActlblHistoryFrame):
        self.frames.append(frame)

    def save_history(self):
        pass # todo implement
