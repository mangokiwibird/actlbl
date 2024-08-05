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

import time


class Timer:
    """Timer Implementation

    Attributes:
        start_time: The start timestamp
        prev_time: Previous timestamp after tick
        ticks_passed: Seconds passed since the start timestamp
        registered_tasks: Tasks registered to be run after a certain amount of time
    """

    def __init__(self):
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.ticks_passed = 0
        self.registered_tasks = {}

    def tick_sec(self):
        """Increase tick every second, executes registered action"""

        current_time = time.time()

        if current_time - self.prev_time > 1:
            self.prev_time = current_time
            self.ticks_passed += 1

            if f"{self.ticks_passed}" in self.registered_tasks:
                while len(self.registered_tasks[f"{self.ticks_passed}"]) > 0:
                    (self.registered_tasks[f"{self.ticks_passed}"].pop())()

    def register_action(self, sec: int, function):
        """registers action to be run on given time

        Args:
            sec: The amount of time needed to be passed for the function to be called
            function: The function to be called
        """

        if f"{sec}" in self.registered_tasks:  # todo: integer keys?
            self.registered_tasks[f"{sec}"].append(function)
        else:
            self.registered_tasks[f"{sec}"] = [function]
