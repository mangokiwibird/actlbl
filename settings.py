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

import os


def is_debug():
    """Returns whether this project is run in debug mode

    Returns:
        Whether this project is run in debug mode, defaulting to False
    """
    return bool(os.environ.get("actlbl_dbg", "False"))


def get_model_path():
    """Returns movenet path

    Returns:
        Movenet Path specified in the environment variable, or a default movenet location
    """
    return str(os.environ.get("actlbl_movenet_path", "./model/movenet_thunder.tflite"))


def get_mjpeg_port():
    """Returns mjpeg server port

    Returns:
        Set MJPEG server port, or a default port
    """
    return int(os.environ.get("actlbl_mjpeg_port", "8080"))


def get_mjpeg_channel_name():
    """Returns mjpeg server port

    Returns:
        Set MJPEG server port, or a default port
    """
    return int(os.environ.get("actlbl_mjpeg_channel_name", "my_camera"))
