import time


class Timer:
    def __init__(self):
        self.result = None
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.result = time.time() - self.start_time
