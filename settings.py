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

def actlbl_configuration(configuration_name):
    configuration_value = str(os.environ.get(configuration_name))
    print(f"{configuration_name}: {configuration_value}")
    return lambda x: x

@actlbl_configuration("actlbl_dbg")
def is_debug():
    """Returns whether this project is run in debug mode

    Returns:
        Whether this project is run in debug mode, defaulting to True
    """
    return bool(os.environ.get("actlbl_dbg", "True"))


@actlbl_configuration("actlbl_movenet_path")
def get_model_path():
    """Returns movenet path

    Returns:
        Movenet Path specified in the environment variable, or a default movenet location
    """
    return str(os.environ.get("actlbl_movenet_path", "./movenet_thunder.tflite"))


@actlbl_configuration("actlbl_mjpeg_port")
def get_mjpeg_port():
    """Returns mjpeg server port

    Returns:
        Set MJPEG server port, or a default port
    """
    return int(os.environ.get("actlbl_mjpeg_port", "8080"))


@actlbl_configuration("actlbl_mjpeg_channel_name")
def get_mjpeg_channel_name():
    """Returns mjpeg server port

    Returns:
        Set MJPEG server port, or a default port
    """
    return str(os.environ.get("actlbl_mjpeg_channel_name", "my_camera"))


@actlbl_configuration("actlbl_target_model_path")
def get_target_model_path():
    """Returns target model path to save after training

    Returns:
        Set target model path, or a default path
    """

    return str(os.environ.get("actlbl_target_model_path", "./model/actlbl_model.keras"))


@actlbl_configuration("actlbl_frames_per_sample")
def get_frames_per_sample():
    """Frames per sample, defaults to 25"""

    return int(os.environ.get("actlbl_frames_per_sample", "32"))


@actlbl_configuration("actlbl_max_frames_in_history")
def get_max_frames_in_history():
    """Max frames in history, defaults to 100"""

    return int(os.environ.get("actlbl_max_frames_in_history", "32"))
