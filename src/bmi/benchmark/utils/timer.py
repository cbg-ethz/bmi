"""Creates a Timer class, a convenient thing to measure the elapsed time."""

import time


class Timer:
    """Class which can be used to measure elapsed experiment time.

    Example:
        >>> timer = Timer()
        >>> timer.reset()
        >>> run_some_command()
        >>> elapsed_time = timer.check()

    Note:
        It is *not* safe to use it together with parallel/concurrent operations.
    """

    def __init__(self) -> None:
        """Initializes and resets the timer."""
        self._t0: float = self._current_time()

    @staticmethod
    def _current_time() -> float:
        """Returns current time."""
        return time.perf_counter()

    def reset(self) -> None:
        """Resets the timer."""
        self._t0 = self._current_time()

    def check(self) -> float:
        """Returns the elapsed time since the last reset."""
        return self._current_time() - self._t0

    def check_and_reset(self) -> float:
        """Checks the timer and resets it.

        For a given timer:
        >>> timer = Timer()
        the expression
        >>> t = timer.check_and_reset()
        is equivalent to
        >>> t = timer.check()
        >>> timer.reset()
        """
        t = self.check()
        self.reset()
        return t
