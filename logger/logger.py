import datetime
import inspect
import logging
import os
import sys
import threading
import wx.lib.newevent

from functools import wraps
from multiprocessing import Queue
from logging.handlers import QueueListener
from typing import Callable

# Define a custom event for logging
LogEvent, EVT_LOG_EVENT = wx.lib.newevent.NewEvent()


def get_function_and_class_name(func: Callable) -> str:
    """
    Get the function and class name of the provided function.

    Args:
        func: The function to extract the name from.

    Returns:
        str: The combined function and class name if applicable.
    """
    # Check if the provided object is a function
    function_name = func.__name__
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        res =  f"{class_name}.{function_name}"
    else:
        res = function_name
    return res


def arg_logger(func: Callable) -> Callable:
    """
    Decorator for logging function arguments and return values.

    Args:
        func: The function to decorate.

    Returns:
        Callable: The decorated function.
    """
    @wraps(func)
    def new_func(*args, **kwargs):
        saved_args = locals()
        logger = get_logger()
        func_name = get_function_and_class_name(func)

        logger.log.debug(f"{func_name}({saved_args})")
        return func(*args, **kwargs)
    return new_func


class LogHandler(logging.Handler):
    """
    Custom logging handler that appends log messages to a wx.TextCtrl widget.

    Args:
        text_ctrl: The wx.TextCtrl widget to append log messages to.
    """
    def __init__(self, text_ctrl):
        logging.Handler.__init__(self)
        self._text_ctrl = text_ctrl

    @property
    def text_ctrl(self):
        return self._text_ctrl

    @text_ctrl.setter
    def text_ctrl(self, value):
        self._text_ctrl = value

    def emit(self, record: logging.LogRecord) -> None:
        """
        Append log message to the text control.
        """
        msg = self.format(record)
        if self.text_ctrl is not None:
            wx.CallAfter(self.text_ctrl.AppendText, msg + '\n')  # Append log message to the text control


class LogFileHandler(logging.FileHandler):
    """
    Custom logging handler that logs into file.

    """
    def __init__(self):
        fp = self.__get_log_file_path()
        super(LogFileHandler, self).__init__(fp)

    @staticmethod
    def __get_log_file_path() -> str:
        """
        Generates the log file path based on the current timestamp.

        Returns:
        str: The log file path.
        """
        folder = os.path.join(sys.path[1], "log")
        os.makedirs(folder, exist_ok=True)
        current_time = datetime.datetime.now()
        file_name = "{}.log".format(current_time.strftime("%y%m%d%H%M"))
        fp = os.path.join(folder, file_name)
        return fp


class Logger:
    """
    Custom logger class that manages logging messages. Singleton class.
    """

    log_level = logging.INFO
    _instance_lock = threading.Lock()
    formatter = None

    def __new__(cls, text_ctrl=None):
        with cls._instance_lock:
            if not hasattr(cls, '_instance'):
                cls._instance = super(Logger, cls).__new__(cls)
                cls._instance.log = logging.getLogger('MyLogger')
                cls._instance.log.setLevel(cls.log_level)
                if cls.log_level == logging.DEBUG:
                    cls.formatter = logging.Formatter(
                        '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                        '%m-%d %H:%M:%S')
                else:
                    cls.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

                gui_handler = LogHandler(text_ctrl)
                gui_handler.setFormatter(cls.formatter)
                cls._instance.log.addHandler(gui_handler)
                console_handler = logging.StreamHandler()
                console_handler.setLevel(cls.log_level)
                console_handler.setFormatter(cls.formatter)
                cls._instance.log.addHandler(console_handler)
                file_handler = LogFileHandler()
                file_handler.setLevel(cls.log_level)
                file_handler.setFormatter(cls.formatter)
                cls._instance.log.addHandler(file_handler)
                cls._instance.text_ctrl = text_ctrl

        return cls._instance

    @property
    def text_ctrl(self):
        return self.log.handlers[0].text_ctrl

    @text_ctrl.setter
    def text_ctrl(self, value):
        self.log.handlers[0].text_ctrl = value

    def start_process_listener(self, queue: Queue) -> None:
        """Start listening for log messages from the given queue. Used to get logs from child processes."""
        self.listener = QueueListener(queue, *self.log.handlers)
        self.listener.start()

    def stop_process_listener(self) -> None:
        """Stop listening for log messages from the queue. Used to stop getting logs from child processes."""
        self.listener.stop()


def get_logger() -> Logger:
    return Logger()
