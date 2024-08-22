from abc import abstractmethod


class Labeler:
    @abstractmethod
    def get_score(self, keypoints):
        pass

    @abstractmethod
    def save_frame(self, keypoints):
        pass

    @abstractmethod
    def save_data(self):
        pass


