""" The stufs that can not be put eleswere. """

DEBUG = True

class DebugPrint: # pylint: disable-next=R0903,
    """Prints the arguments if debug_state is True."""
    def __init__(self, debug_state: bool = DEBUG):
        """Initializes the DebugPrint class."""
        self.debug_state = debug_state

    def __call__(self, *args, **kwargs):
        """Prints the arguments if debug_state is True."""
        if self.debug_state:
            print(*args, **kwargs)

debug_print = DebugPrint()
