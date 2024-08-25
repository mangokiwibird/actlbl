import os

import settings
from camera import start_fs_capture
from image import LabeledImage
from labeler.ml_model import load_dataset, train_labeler


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


def generate_model_fs():
    classified_history = [
        load_dataset("./model/running"),
        load_dataset("./model/lying"),
        load_dataset("./model/walking"),
        load_dataset("./model/sitting"),
        load_dataset("./model/standing")
    ]

    train_labeler(classified_history)


def test_from_fs(video_path, model_path):
    def callback(img: LabeledImage):
        img.get_activity()
        if "counter" in img.camera_context.settings:
            img.camera_context.settings.counter += 1
        else:
            img.camera_context.settings.counter = 1

    ctx = start_fs_capture(video_path, callback, target_path="model_test_results.json", model_path=model_path)

    if settings.is_debug():
        print(f"Frames in current video: {ctx.settings.get("counter")}")


# TODO: remove hardcoding
def record_from_fs(dataset_directory="dataset"):
    video_map = classify_videos(dataset_directory)

    min_frames = 10000

    for activity, video_paths in video_map.items():
        for video_path in video_paths:
            print(f"given: {activity}")

            def callback(img: LabeledImage):
                img.record_activity()
                if "counter" in img.camera_context.settings:
                    img.camera_context.settings["counter"] += 1
                else:
                    img.camera_context.settings["counter"] = 1

            ctx = start_fs_capture(video_path, callback, target_path=f"model/{activity}/{video_path.split("dataset\\")[1]}.json")

            if settings.is_debug():
                current_frames = int(ctx.settings.get("counter"))

                if min_frames > current_frames:
                    min_frames = current_frames
                print(f"Frames in current video: {current_frames}")

    print(f"Min Frames: {min_frames}")

