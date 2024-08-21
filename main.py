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

from camera import start_local_capture
from movenet import load_model
from fs_training import train_from_fs
from local_training import train_from_local

parser = argparse.ArgumentParser(
                    prog='ACTLBL',
                    description='Motion Parser for RnE')

parser.add_argument('--record', type=bool, action='store_true')
parser.add_argument("--test_model", type=bool, action="store_true")

args = parser.parse_args()

load_model()  # Load movenet model

print("------NOTICE------")
print("Check settings.py for configuration")
print("------------------")

if args.record:
    train_from_local()
elif args.test_model:
    train_from_fs()
else:
    print("No valid action passed. Possible options: --record or --test_model")
