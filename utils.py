DEBUG = True

class DebugPrint:
    def __init__(self, debug_state: bool = DEBUG):
        self.debug_state = debug_state

    def __call__(self, *args, **kwargs):
        if self.debug_state:
            print(*args, **kwargs)

debug_print = DebugPrint()
