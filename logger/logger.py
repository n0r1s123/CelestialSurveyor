import datetime
import inspect
import logging
import os
import sys
import wx.lib.newevent
import threading

# Define a custom event for logging
LogEvent, EVT_LOG_EVENT = wx.lib.newevent.NewEvent()


def get_logger():
    return Logger()


from functools import wraps


def get_function_and_class_name(func):
    # Check if the provided object is a function

    function_name = func.__name__
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        res =  f"{class_name}.{function_name}"
    else:
        res = function_name
    return res


def arg_logger(func):

    @wraps(func)
    def new_func(*args, **kwargs):
        saved_args = locals()
        logger = get_logger()
        func_name = get_function_and_class_name(func)

        logger.log.debug(f"{func_name}({saved_args})")
        return func(*args, **kwargs)

    return new_func


class LogHandler(logging.Handler):
    def __init__(self, text_ctrl):
        logging.Handler.__init__(self)
        self._text_ctrl = text_ctrl

    @property
    def text_ctrl(self):
        return self._text_ctrl

    @text_ctrl.setter
    def text_ctrl(self, value):
        self._text_ctrl = value

    def emit(self, record):
        msg = self.format(record)
        if self.text_ctrl is not None:
            wx.CallAfter(self.text_ctrl.AppendText, msg + '\n')  # Append log message to the text control

class LogFileHandler(logging.FileHandler):
    def __init__(self):
        fp = self.__get_log_file_path()
        super(LogFileHandler, self).__init__(fp)

    @staticmethod
    def __get_log_file_path():
        folder = os.path.join(sys.path[1], "log")
        os.makedirs(folder, exist_ok=True)
        current_time = datetime.datetime.now()
        file_name = "{}.log".format(current_time.strftime("%y%m%d%H%M"))
        fp = os.path.join(folder, file_name)
        return fp


class Logger:
    log_level = logging.INFO
    _instance_lock = threading.Lock()

    def __new__(cls, text_ctrl=None):
        with cls._instance_lock:
            if not hasattr(cls, '_instance'):
                cls._instance = super(Logger, cls).__new__(cls)
                cls._instance.log = logging.getLogger('MyLogger')
                cls._instance.log.setLevel(cls.log_level)

                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

                # Create a handler to add log messages to the text control
                gui_handler = LogHandler(text_ctrl)
                gui_handler.setFormatter(formatter)
                cls._instance.log.addHandler(gui_handler)

                # Create another handler to log messages to the console
                console_handler = logging.StreamHandler()
                console_handler.setLevel(cls.log_level)
                console_handler.setFormatter(formatter)
                cls._instance.log.addHandler(console_handler)
                file_handler = LogFileHandler()
                file_handler.setLevel(cls.log_level)
                file_handler.setFormatter(formatter)
                cls._instance.log.addHandler(file_handler)
                cls._instance.text_ctrl = text_ctrl

        return cls._instance

    @property
    def text_ctrl(self):
        return self.log.handlers[0].text_ctrl

    @text_ctrl.setter
    def text_ctrl(self, value):
        self.log.handlers[0].text_ctrl = value
