import tqdm

from abc import ABC, abstractmethod
from wx import Gauge


class AbstractProgressBar(ABC):
    @abstractmethod
    def update(self, num: int = 1):
        """
        Update the progress bar by the given number of units.
        :param num: An integer representing the number of units to update.
        """
        pass

    @abstractmethod
    def complete(self):
        """
        Mark the progress bar as complete.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear the progress bar from the console or user interface.
        """
        pass

    @abstractmethod
    def set_description(self, description: str):
        """
        Set a description for the progress bar.
        :param description: A string describing the progress bar.
        """
        pass

    @abstractmethod
    def set_total(self, total: int):
        """
        Set the total number of units for the progress bar.
        :param total: An integer representing the total number of units.
        """
        pass


class ProgressBarCli(AbstractProgressBar):
    """
    This class represents a command-line progress bar.
    """
    def __init__(self):

        self.progress_bar = None

    def update(self, num: int = 1):

        self.progress_bar.update()

    def complete(self):
        self.progress_bar.close()
        self.progress_bar.clear()
        self.progress_bar = None

    def clear(self):
        self.progress_bar.clear()

    def set_description(self, description: str):
        self.progress_bar.set_description(description)

    def set_total(self, total: int):
        self.progress_bar = tqdm.tqdm(total=total)
        self.progress_bar.display()

    def _draw(self):
        pass


class ProgressBarGui(AbstractProgressBar):
    """
    This class represents a WxPython progress bar used in UI.
    """
    def __init__(self, progress_bar: Gauge):
        self.progress_bar = progress_bar
        self.progress_bar.SetValue(0)

    def update(self, num: int = 1):
        self.progress_bar.SetValue(self.progress_bar.GetValue() + num)

    def complete(self):
        pass

    def clear(self):
        self.progress_bar.SetValue(0)

    def set_description(self, description: str):
        pass

    def set_total(self, total: int):
        self.progress_bar.SetRange(total)

    def _draw(self):
        pass


class ProgressBarFactory:
    """
    A factory class for creating different types of progress bars.
    """
    @staticmethod
    def create_progress_bar(progress_bar_instance) -> AbstractProgressBar:
        """
        Create and return a specific type of progress bar based on the given instance.

        Args:
            progress_bar_instance (SharedMemoryParams): An instance of a progress bar.

        Returns:
            tAbstractProgressBar: An AbstractProgressBar instance representing a specific type of progress bar.
        """
        if isinstance(progress_bar_instance, tqdm.tqdm):
            return ProgressBarCli()
        elif isinstance(progress_bar_instance, Gauge):
            return ProgressBarGui(progress_bar_instance)
        else:
            raise ValueError("Invalid progress bar type")
