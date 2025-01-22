import time

import cv2


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

    # TODO: better naming
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

    def display_timer(self, image):
        """Adds timer information to the upper right corner of the image in text

        Args:
            image: Image to add timer info onto

        Returns:
             Synthesized image with time displayed on the left top corner
        """

        return cv2.putText(image, f"{self.ticks_passed}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))