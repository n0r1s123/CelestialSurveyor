import logging
import wx.lib.newevent
import threading

# Define a custom event for logging
LogEvent, EVT_LOG_EVENT = wx.lib.newevent.NewEvent()


class LogHandler(logging.Handler):
    def __init__(self, text_ctrl):
        logging.Handler.__init__(self)
        self.text_ctrl = text_ctrl

    def emit(self, record):
        msg = self.format(record)
        if self.text_ctrl is not None:
            wx.CallAfter(self.text_ctrl.AppendText, msg + '\n')  # Append log message to the text control


class Logger:
    _instance_lock = threading.Lock()

    def __new__(cls, text_ctrl=None):
        with cls._instance_lock:
            if not hasattr(cls, '_instance'):
                cls._instance = super(Logger, cls).__new__(cls)
                cls._instance.log = logging.getLogger('MyLogger')
                cls._instance.log.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

                # Create a handler to add log messages to the text control
                gui_handler = LogHandler(text_ctrl)
                gui_handler.setFormatter(formatter)
                cls._instance.log.addHandler(gui_handler)

                # Create another handler to log messages to the console
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(formatter)
                cls._instance.log.addHandler(console_handler)

        return cls._instance
