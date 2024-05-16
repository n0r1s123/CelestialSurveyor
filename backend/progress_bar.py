import tqdm

from abc import ABC, abstractmethod
from wx import Gauge
from typing import Optional


class AbstractProgressBar(ABC):
    @abstractmethod
    def update(self, num: int = 1):
        """
        Update the progress bar with a given progress value.
        :param progress: A float value between 0 and 1 indicating the progress.
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
    def __init__(self):

        self.progress_bar = None

    def update(self, num: int = 1):

        self.progress_bar.update()

    def complete(self):
        # self.progress_bar.refresh()
        self.progress_bar.close()
        self.progress_bar.clear()
        self.progress_bar = None

    def clear(self):
        self.progress_bar.clear()

    def set_description(self, description: str):
        # Implement description setting logic specific to ProgressBar1
        self.progress_bar.set_description(description)

    def set_total(self, total: int):
        self.progress_bar = tqdm.tqdm(total=total)
        # self.progress_bar.total = total
        self.progress_bar.display()

    def _draw(self):
        pass


class ProgressBarGui(AbstractProgressBar):
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
    @staticmethod
    def create_progress_bar(progress_bar_instance) -> AbstractProgressBar:
        if isinstance(progress_bar_instance, tqdm.tqdm):
            return ProgressBarCli(progress_bar_instance)
        elif isinstance(progress_bar_instance, Gauge):
            return ProgressBarGui(progress_bar_instance)
        else:
            raise ValueError("Invalid progress bar type")
