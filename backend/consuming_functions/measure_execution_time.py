import time

from logger.logger import get_logger

logger = get_logger()


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.log.info(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper
