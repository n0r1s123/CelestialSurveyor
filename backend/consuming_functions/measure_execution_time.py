import time

from logger.logger import get_logger
from typing import Callable, Any


logger = get_logger()


def measure_execution_time(func: Callable) -> Callable:
    """
    A decorator function to measure the execution time of a given function.

    Args:
        func (callable): The function to measure the execution time for.

    Returns:
        callable: The wrapped function that measures the execution time.
    """
    def wrapper(*args, **kwargs) -> Any:
        """
        Wrapper function to measure the execution time of the decorated function.

        Args:
            *args: Positional arguments for the decorated function.
            **kwargs: Keyword arguments for the decorated function.

        Returns:
            Any: The result of the decorated function.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.log.info(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper
