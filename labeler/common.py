from abc import abstractmethod

from frame_loader import FrameLoader


class Labeler(FrameLoader):
    @abstractmethod
    def get_score(self, keypoints):
        pass
