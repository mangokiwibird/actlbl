#     Copyright (C) 2024 dolphin2410
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
import argparse

import settings
from camera import start_capture
from image import LabeledImage
from labeler.ml_model import generate_model
from movenet_wrapper import load_movenet_model
from dataset_manager import extract_keypoints

parser = argparse.ArgumentParser(
                    prog='ACTLBL',
                    description='Motion Parser for RnE')

parser.add_argument('--record_live', action='store_true')
parser.add_argument("--mp4_to_keypoints", action="store_true")
parser.add_argument('--test_from_video', action='store_true')
parser.add_argument('--generate_model', action='store_true')

parser.add_argument('--video_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--dataset_path', type=str)

args = parser.parse_args()

load_movenet_model()  # Load movenet model

print("------NOTICE------")
print("Check settings.py for configuration")
print("------------------")

# Test pretrained model with the given video
# Usage: python main.py --test_keypoint_graph --video_path <video_path> --model_path <model_path>
if args.test_from_video and args.video_path and args.model_path:
    # TODO test_from_fs(args.video_path, args.model_path)
    print("Implementation required")

# Preprocess video into keypoint graph
# Usage: python main.py --record_from_mp4 --video_path <video_path>
elif args.mp4_to_keypoints and args.video_path:
    print(extract_keypoints(args.video_path)) # TODO save to file

# Preprocess live video into keypoint graph
# Usage: python main.py --record_live
elif args.record_live:
    def callback(labeled_img: LabeledImage):
        labeled_img.camera_context.ml_labeler.save_frame(labeled_img.keypoints) # TODO create a new frame saver

    def exit_callback(labeled_img: LabeledImage):
        history = labeled_img.camera_context.ml_labeler.get_data()
        print(history) # TODO save this to a file

    ctx = start_capture(0, callback, None, exit_callback)


# Train ML model from videos from the given dataset path
# Usage: python main.py --generate_model --dataset_path <dataset_path>
elif args.generate_model and args.dataset_path:
    generate_model(args.dataset_path)

else:
    print("No valid action passed")
