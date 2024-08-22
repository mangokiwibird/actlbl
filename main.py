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

from movenet import load_model
from fs_training import record_from_fs, test_from_fs
from local_training import record_from_local

parser = argparse.ArgumentParser(
                    prog='ACTLBL',
                    description='Motion Parser for RnE')

parser.add_argument('--record_live', action='store_true')
parser.add_argument("--test_model", action="store_true")
parser.add_argument("--record_from_mp4", action="store_true")
parser.add_argument('--test_from_mp4', action='store_true')
parser.add_argument('--video_path', type=str)
parser.add_argument('--model_path', type=str)

args = parser.parse_args()

load_model()  # Load movenet model

print("------NOTICE------")
print("Check settings.py for configuration")
print("------------------")

if args.test_from_mp4 and args.video_path and args.model_path:
    test_from_fs(args.video_path, args.model_path)
elif args.record_from_mp4 and args.video_path:
    record_from_fs(args.video_path)
elif args.record_live:
    record_from_local()
elif args.test_model:
    record_from_fs()
else:
    print("No valid action passed. Possible options: --record or --test_model")
