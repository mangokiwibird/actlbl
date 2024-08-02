import time

class Timer:
    """Timer Implementation"""

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
        """registers action to be run on given time"""

        if f"{sec}" in self.registered_tasks: # todo: integer keys?
            self.registered_tasks[f"{sec}"].append(function)
        else:
            self.registered_tasks[f"{sec}"] = [function]