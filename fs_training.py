import os

from camera import start_fs_capture
from image import LabeledImage


# TODO this code relies on the -ing suffix of an activity <sort of hardcoding?>
def parse_video_path(video_file: str):
    raw_split_video_path = video_file.split(".mp4")[0].split("ing")

    # validate file name else return None
    if len(raw_split_video_path) != 2 or not raw_split_video_path[1].isdigit():
        return None

    activity_name = raw_split_video_path[0] + "ing"  # standing sitting walking running lying
    return activity_name


def classify_videos(directory: str):
    """
    classifies video paths by its name
    """
    list_videos = {"running": [], "sitting": [], "walking": [], "standing": [], "lying": []}

    files = os.listdir(directory)
    for video_file in files:
        print(video_file)
        video_activity = parse_video_path(video_file)
        if video_activity is None:
            print("non????")
            continue

        if video_activity not in list_videos:
            raise Exception("unregistered activity!")

        full_path = os.path.join(directory, video_file)

        list_videos[video_activity].append(full_path)

    return list_videos


# TODO: remove hardcoding
def train_from_fs():
    video_map = classify_videos("dataset")

    for activity, video_paths in video_map.items():
        for video_path in video_paths:
            print(f"given: {activity}")

            def callback(img: LabeledImage):
                print(img.get_activity())

            start_fs_capture(video_path, callback, target_path=f"model/{activity}/{video_path.split("dataset\\")[1]}.json")
