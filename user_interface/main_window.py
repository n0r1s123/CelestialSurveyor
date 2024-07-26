import datetime
import json
import numpy as np
import os.path
import threading
import wx
import wx.lib.scrolledpanel as scrolled

from dataclasses import dataclass
from decimal import Decimal
from ObjectListView import ObjectListView, ColumnDefn
from PIL import Image
from threading import Event
from typing import Optional, Callable, Any

from backend.find_asteroids import predict_asteroids, save_results, annotate_results
from backend.progress_bar import ProgressBarFactory
from backend.source_data_v2 import SourceDataV2
from logger.logger import get_logger


logger = get_logger()


@dataclass(frozen=True)
class MyDataObject:
    """
    Represents an object within file list.

    Attributes:
        file_path (str): The path to the file.
        timestamp (datetime.datetime): The timestamp of the object.
        exposure (Decimal): The exposure value of the object.
        checked (bool): The status of the object if it is checked or not in the list, default is False.
    """
    file_path: str
    timestamp: datetime.datetime
    exposure: Decimal
    checked: bool = False


class ImagePanel(scrolled.ScrolledPanel):
    """
    A panel for displaying images with zoom and scrolling capability.
    Note: Scrolling doesn't work for now.
    """
    def __init__(self, parent: wx.Window, image_array: Optional[np.ndarray] = None):
        super().__init__(parent, style=wx.BORDER_SIMPLE)
        self.image_array = image_array
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_scroll)
        self.scale_factor = None
        self.default_scale_factor = 1.0
        self.scroll_x = 0
        self.scroll_y = 0
        self.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)

    def on_paint(self, event: Event) -> None:
        """
        Callback function for when the panel needs to paint its content.

        Parameters:
            event (wx.Event): A wxPython event object.

        Returns:
            None
        """
        if self.image_array is not None:
            display_width, display_height = self.GetSize()
            image_height, image_width = self.image_array.shape[:2]
            dc = wx.PaintDC(self)
            if self.scale_factor is None:
                self.scale_factor = display_width / image_width
                self.default_scale_factor = self.scale_factor

            dc.SetUserScale(self.scale_factor, self.scale_factor)
            image = self.convert_array_to_bitmap(self.image_array)
            dc.DrawBitmap(image, 0, 0)

    def on_scroll(self, event: Event) -> None:
        """
        Callback function for when the mouse wheel is scrolled.
        It's zooming.

        Parameters:
            event (wx.Event): A wxPython event object.

        Returns:
            None
        """

        rotation = event.GetWheelRotation()

        # Calculate the new scale factor based on the mouse wheel rotation
        if rotation > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1

        self.Refresh(eraseBackground=False)

    @staticmethod
    def convert_array_to_bitmap(array: np.ndarray, display_width: int = None, display_height: int = None) -> wx.Bitmap:
        """
        Convert a NumPy array to a wx.Bitmap to be displayed on the image panel.

        Parameters:
            array (np.ndarray): The input NumPy array.
            display_width (int): The width for displaying the bitmap.
            display_height (int): The height for displaying the bitmap.

        Returns:
            wx.Bitmap: The converted wx.Bitmap object.
        """
        # Create a PIL Image from the NumPy array
        array = array.reshape(array.shape[:2])
        image = Image.fromarray(array)

        # Convert PIL Image to wx.Image
        pil_image = image.convert("RGB")
        if display_width is not None and display_height is not None:
            pil_image = pil_image.resize((display_width, display_height))

        width, height = pil_image.size
        image = wx.Image(width, height)
        image.SetData(pil_image.tobytes())

        # Convert wx.Image to wx.Bitmap
        bitmap = image.ConvertToBitmap()

        return bitmap


class ProgressFrame(wx.Dialog):
    """Progress dialog to display progress of the current operation"""

    def __init__(self, parent: wx.Frame, title: str, stop_event: Event):
        super().__init__(parent=parent, title=title, size=(500, 200))
        self.parent = parent
        self.stop_event = stop_event
        panel = wx.Panel(self)
        self.label = wx.StaticText(panel, label="Working...")
        self.progress = wx.Gauge(panel, range=100, size=(450, 30))
        self.cancel_button = wx.Button(panel, label='Cancel')
        self.Bind(wx.EVT_BUTTON, self.on_cancel, self.cancel_button)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.label, 0, wx.ALL | wx.LEFT, 5)
        sizer.Add(self.progress, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.cancel_button, 0, wx.ALL | wx.CENTER, 5)
        panel.SetSizer(sizer)

        self.failed = False

    def set_failed(self, value: bool) -> None:
        """
        Sets the 'failed' attribute of the ProgressFrame instance.
        Used to indicate that the operation has failed and display an error message.

        Args:
            value (bool): The value to set the 'failed' attribute to.

        Returns:
            None
        """
        self.failed = value

    def on_cancel(self, event: Event) -> None:
        """
        Callback function for when the cancel button is clicked.
        It's stopping the current operation.

        Parameters:
            event (wx.Event): A wxPython event object.

        Returns:
            None
        """
        if self.failed:
            self.Close(force=True)
            self.set_failed(False)
        else:
            self.label.SetLabel("Stopping...")
            self.stop_event.set()


class MyFrame(wx.Frame):
    """Main frame of the application"""

    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)

        self.source_data: SourceDataV2 = None

        # Create the main panel
        self.panel = wx.Panel(self)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)

        # Create a horizontal box sizer to organize the elements
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        # Create the controls area (1/6 of window width)
        controls_sizer = wx.BoxSizer(wx.VERTICAL)

        controls_label = wx.StaticText(self.panel)
        flat_label = wx.StaticText(self.panel, label="Select folder with flat frames")
        self.flat_path_picker = wx.DirPickerCtrl(self.panel, style=wx.DIRP_USE_TEXTCTRL)
        dark_flat_label = wx.StaticText(self.panel, label="Select folder with dark flat frames")
        self.dark_flat_path_picker = wx.DirPickerCtrl(self.panel, style=wx.DIRP_USE_TEXTCTRL)
        dark_label = wx.StaticText(self.panel, label="Select folder with dark frames")
        self.dark_path_picker = wx.DirPickerCtrl(self.panel, style=wx.DIRP_USE_TEXTCTRL)
        self.btn_add_files = wx.Button(self.panel, label="Add files")
        self.Bind(wx.EVT_BUTTON, self.on_add_files, self.btn_add_files)
        self.chk_to_align = wx.CheckBox(self.panel, label="Align images")
        self.chk_to_align.SetValue(True)
        self.chk_non_linear = wx.CheckBox(self.panel, label="Non-linear")
        self.chk_non_linear.SetValue(False)
        self.to_debayer = wx.CheckBox(self.panel, label="Debayer")
        self.to_debayer.SetValue(True)
        self.btn_load_files = wx.Button(self.panel, label="Load files")
        self.progress_frame = None
        self.process_thread = None
        self.stop_event = None

        self.Bind(wx.EVT_BUTTON, self.on_load_files, self.btn_load_files)
        self.results_label = wx.StaticText(self.panel, label="Select folder to store results")
        self.results_path_picker = wx.DirPickerCtrl(self.panel, style=wx.DIRP_USE_TEXTCTRL)
        self.magnitude_label = wx.StaticText(self.panel, label="Annotation magnitude limit")
        self.magnitude_input = wx.TextCtrl(self.panel)
        self.magnitude_input.SetValue("18.0")
        self.magnitude_input.Bind(wx.EVT_TEXT, self.on_text_change)
        self.btn_process = wx.Button(self.panel, label="Process")
        self.Bind(wx.EVT_BUTTON, self.on_process, self.btn_process)
        self.btn_start_again = wx.Button(self.panel, label="Start again")
        self.Bind(wx.EVT_BUTTON, self.on_start_again, self.btn_start_again)

        controls_sizer.Add(controls_label, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(flat_label, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.flat_path_picker, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(dark_flat_label, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.dark_flat_path_picker, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(dark_label, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.dark_path_picker, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.btn_add_files, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.to_debayer, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.chk_to_align, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.chk_non_linear, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.btn_load_files, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.results_label, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.results_path_picker, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.magnitude_label, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.magnitude_input, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.btn_process, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.btn_start_again, 0, wx.EXPAND | wx.ALL, 5)

        hbox.Add(controls_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Create the checkbox list area (2/6 of window width) using ObjectListView
        checkbox_label = wx.StaticText(self.panel)
        self.checkbox_list = ObjectListView(self.panel, style=wx.LC_REPORT | wx.SUNKEN_BORDER)
        self.checkbox_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_item_selected)
        columns = [
            ColumnDefn("Item", "left", 350, "file_path"),
            ColumnDefn("Timestamp", "left", 130, "timestamp"),
            ColumnDefn("Exposure", "left", 60, "exposure"),
        ]

        self.checkbox_list.SetColumns(columns)
        self.checkbox_list.CreateCheckStateColumn()
        hbox.Add(checkbox_label, 0, wx.EXPAND | wx.ALL, 5)
        hbox.Add(self.checkbox_list, 2, wx.EXPAND | wx.ALL, 5)

        # Create the picture representation area (3/6 of window width)
        self.draw_panel = ImagePanel(self.panel)
        hbox.Add(self.draw_panel, 3, wx.EXPAND | wx.ALL, 5)

        self.log_text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        logger.text_ctrl = self.log_text
        panel_sizer.Add(hbox, 3, wx.EXPAND | wx.ALL, 5)
        panel_sizer.Add(self.log_text, 1, wx.EXPAND | wx.ALL, 5)

        # Set the sizer for the main panel
        self.panel.SetSizer(panel_sizer)

        # Create the menu bar
        menubar = wx.MenuBar()

        # Create a File menu
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_EXIT, "Exit", "Exit the application")
        menubar.Append(file_menu, "&File")
        self.set_startup_states()

        # Set the menu bar for the frame
        self.SetMenuBar(menubar)
        # Set the frame properties
        self.SetSize((1900, 1000))
        self.SetTitle("CelestialSurveyor")
        self.Centre()
        self.Maximize(True)

        # Bind events
        self.Bind(wx.EVT_MENU, self.on_exit, id=wx.ID_EXIT)
        logger.log.info("App initialized")

    def on_text_change(self, event: Event) -> None:
        """
        Event handler for text change in the magnitude limit input field.
        If the input value is greater than 25, an error message is shown.

        Args:
            event: The event object that triggered this function.

        Returns:
            None
        """
        input_value = self.magnitude_input.GetValue()
        try:
            float_value = float(input_value)
            if float_value > 25:
                wx.MessageBox("Input value must be less than 25", "Error", wx.OK | wx.ICON_ERROR)
        except ValueError:
            wx.MessageBox("Input value must be a float", "Error", wx.OK | wx.ICON_ERROR)

    @staticmethod
    def __get_previous_settings() -> dict:
        """
        Retrieves the previous settings from the 'settings.json' file.
        Used to populate calibration folder paths. It's too annoying to type them every time.

        Returns:
            dict: The previous settings loaded from the file, or an empty dictionary if the file does not exist.
        """
        settings_file = "settings.json"
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as fileo:
                return json.load(fileo)
        else:
            return {}

    @staticmethod
    def __write_previous_settings(settings: dict) -> None:
        """Writes the previous settings to the 'settings.json' file.

        Args:
            settings (dict): The settings to be written to the file.

        Returns:
            None
        """
        with open("settings.json", 'w') as fileo:
            json.dump(settings, fileo)

    def set_startup_states(self):
        """
        This function sets the startup states of the UI elements taking in account the previous settings.
        """
        prev_settings = self.__get_previous_settings()
        dark_path = prev_settings.get('dark_path', '')
        dark_path = "" if dark_path is None else dark_path
        self.dark_path_picker.SetPath(dark_path)
        flat_path = prev_settings.get('flat_path', '')
        flat_path = "" if flat_path is None else flat_path
        self.flat_path_picker.SetPath(flat_path)
        dark_flat_path = prev_settings.get('dark_flat_path', '')
        dark_flat_path = "" if dark_flat_path is None else dark_flat_path
        self.dark_flat_path_picker.SetPath(dark_flat_path)
        self.chk_to_align.Enable(False)
        self.to_debayer.Enable(False)
        self.chk_non_linear.Enable(False)
        self.btn_load_files.Enable(False)
        self.results_path_picker.Enable(False)
        self.results_label.Enable(False)
        self.magnitude_label.Enable(False)
        self.magnitude_input.Enable(False)
        self.btn_process.Enable(False)
        self.checkbox_list.SetObjects([])
        self.checkbox_list.Refresh(eraseBackground=False)
        self.results_path_picker.SetPath('')

    def on_exit(self, event: Event) -> None:
        """
        Event handler for the exit button.

        Args:
            event: The event object that triggered this function.

        Returns:
            None
        """
        logger.log.info("App exiting")
        self.Close()

    @staticmethod
    def _gen_short_file_names(file_list: list[str]) -> list[str]:
        """
        Generates short file names from a list of full file paths.
        Cuts off the common part of the file paths to make representation shorter.

        Args:
            file_list (list[str]): List of full file paths.

        Returns:
            list[str]: List of short file names.
        """
        folders = {os.path.split(fp)[0] for fp in file_list}
        split = os.path.split
        if len(folders) == 1:
            short_file_names = [split(fp)[1] for fp in file_list]
        else:
            short_file_names = [os.path.join(split(split(fp)[0])[1], split(fp)[1]) for fp in file_list]
        return short_file_names

    def on_header_loaded(self) -> None:
        """
        Event handler for the header loaded event.
        Displays the header information in the file list.
        Enables elements responsible for the next step: image loading, alignment and processing.

        Returns:
            None
        """
        self.source_data.stop_event.clear()

        self.checkbox_list.SetObjects([])
        short_file_paths = self._gen_short_file_names([item.file_name for item in self.source_data.headers])
        self.checkbox_list.SetObjects([
            MyDataObject(
                fp, header.timestamp, header.exposure, checked=True
            ) for fp, header in zip(short_file_paths, self.source_data.headers)
        ])
        objects = self.checkbox_list.GetObjects()
        for obj in objects:
            self.checkbox_list.SetCheckState(obj, True)
        self.checkbox_list.RefreshObjects(objects)
        paths = [item.file_name for item in self.source_data.headers]
        logger.log.debug(f"Added the following files: {paths}")
        if paths:
            self.btn_load_files.Enable(True)
            self.chk_to_align.Enable(True)
            self.to_debayer.Enable(True)
            self.chk_non_linear.Enable(True)

    def handle_load_error(self, func: Callable, progress_frame, message: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            progress_bar = kwargs.get("progress_bar")
            if progress_bar:
                progress_frame.label.SetLabel(message)
                progress_frame.label.Update()
                progress_frame.progress.SetValue(0)
                progress_frame.progress.SetRange(0)
                progress_frame.set_failed(True)
            raise

    def on_add_files(self, event: Event) -> None:
        """
        Event handler for the add files button.

        Opens dialog to select FIT(s) or XISF files.

        Args:
            event: The event object that triggered this function.

        Returns:
            None
        """
        wildcard = "Fits files (*.fits)|*.fits;*.FITS;*.fit;*.FIT|XISF files (*.xisf)|*.xisf;*.XISF|All files (*.*)|*.*"

        dialog = wx.FileDialog(self, message="Choose files to load",
                               wildcard=wildcard,
                               style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)

        paths = []
        if dialog.ShowModal() == wx.ID_OK:
            paths = dialog.GetPaths()
            if self.source_data is None:
                self.source_data: SourceDataV2 = SourceDataV2()

            def load_headers():
                progress_frame = ProgressFrame(self, "Loading headers...", stop_event=self.source_data.stop_event)
                progress_frame.Show()
                self.handle_load_error(
                    self.source_data.extend_headers, progress_frame, "Failed to load headers", file_list=paths,
                    progress_bar=ProgressBarFactory.create_progress_bar(progress_frame.progress))
                progress_frame.Close()
                wx.CallAfter(self.on_header_loaded)
            self.process_thread = threading.Thread(target=load_headers)

            self.process_thread.start()
        dialog.Destroy()

    def on_load_files(self, event: Event) -> None:
        """
        Event handler for loading files.
        Performs loading images, calibration, alignment, cropping and further processing.
        """
        selected_headers = []
        objects = self.checkbox_list.GetObjects()
        for header, obj in zip(self.source_data.headers, objects):
            checked = self.checkbox_list.GetCheckState(obj)
            if checked:
                selected_headers.append(header)
            else:
                logger.log.debug(f"Excluding {header.file_name}")
        self.source_data.set_headers(selected_headers)
        short_file_paths = self._gen_short_file_names([item.file_name for item in self.source_data.headers])
        self.checkbox_list.SetObjects([
            MyDataObject(
                fp, header.timestamp, header.exposure, checked=True
            ) for fp, header in zip(short_file_paths, self.source_data.headers)
        ])
        objects = self.checkbox_list.GetObjects()
        for obj in objects:
            self.checkbox_list.SetCheckState(obj, True)
        self.checkbox_list.RefreshObjects(objects)

        to_align = self.chk_to_align.GetValue()

        logger.log.info(f"Loading {len(self.source_data.headers)} images...")
        dark_path = self.dark_path_picker.GetPath()
        dark_path = dark_path if dark_path else None
        flat_path = self.flat_path_picker.GetPath()
        flat_path = flat_path if flat_path else None
        dark_flat_path = self.dark_flat_path_picker.GetPath()
        dark_flat_path = dark_flat_path if dark_flat_path else None
        self.__write_previous_settings({
            'dark_path': dark_path,
            'flat_path': flat_path,
            'dark_flat_path': dark_flat_path
        })
        self.source_data.to_debayer = self.to_debayer.GetValue()
        self.source_data.linear = not self.chk_non_linear.GetValue()
        self.stop_event = Event()

        def load_images_calibrate_and_align() -> None:
            """
            This function loads, calibrates, and aligns images.
            It's run within the thread.
            """
            progress_frame = ProgressFrame(self, "Loading progress", stop_event=self.source_data.stop_event)
            progress_frame.progress.SetValue(0)
            progress_frame.label.SetLabel("Loading images...")
            progress_frame.Show()
            self.handle_load_error(
                self.source_data.load_images, progress_frame, "Failed to load images",
                progress_bar=ProgressBarFactory.create_progress_bar(progress_frame.progress))
            master_dark = None
            master_dark_flat = None
            master_flat = None
            if not self.source_data.stop_event.is_set():
                if dark_path:
                    if os.path.exists(dark_path):
                        progress_frame.label.SetLabel("Loading dark...")
                        master_dark = self.handle_load_error(
                            self.source_data.make_master_dark, progress_frame, "Failed to make master dark",
                            self.source_data.make_file_paths(dark_path),
                            progress_bar=ProgressBarFactory.create_progress_bar(progress_frame.progress))
                    else:
                        logger.log.warning(f"Dark path {dark_path} does not exist. Skipping make master dark...")
            if not self.source_data.stop_event.is_set():
                if dark_flat_path:
                    if os.path.exists(dark_flat_path):
                        progress_frame.label.SetLabel("Loading dark flat...")
                        master_dark_flat = self.handle_load_error(
                            self.source_data.make_master_dark, progress_frame, "Failed to make master dark flat",
                            self.source_data.make_file_paths(dark_flat_path),
                            progress_bar=ProgressBarFactory.create_progress_bar(progress_frame.progress))
                    else:
                        logger.log.warning(f"Dark flat path {dark_flat_path} does not exist. "
                                           f"Skipping make master dark flat...")

            if not self.source_data.stop_event.is_set():
                if flat_path:
                    if os.path.exists(flat_path):
                        progress_frame.label.SetLabel("Loading flats...")
                        flats = self.handle_load_error(
                            self.source_data.load_flats, progress_frame, "Failed to load flats",
                            self.source_data.make_file_paths(flat_path),
                            progress_bar=ProgressBarFactory.create_progress_bar(progress_frame.progress))
                        if master_dark_flat is not None:
                            for flat in flats:
                                flat -= master_dark_flat
                        master_flat = np.average(flats, axis=0)
            if master_dark is not None:
                self.source_data.original_frames -= master_dark
            if master_flat is not None:
                self.source_data.original_frames /= master_flat
            if to_align and not self.source_data.stop_event.is_set():
                progress_frame.label.SetLabel("Plate solving...")
                self.handle_load_error(
                    self.source_data.plate_solve_all, progress_frame, "Failed to plate solve images",
                    progress_bar=ProgressBarFactory.create_progress_bar(progress_frame.progress))
            if to_align and not self.source_data.stop_event.is_set():
                progress_frame.label.SetLabel("Aligning images...")
                self.handle_load_error(
                    self.source_data.align_images_wcs, progress_frame, "Failed to align images",
                    progress_bar=ProgressBarFactory.create_progress_bar(progress_frame.progress))
            if not self.source_data.stop_event.is_set():
                progress_frame.label.SetLabel("Cropping images...")
                self.handle_load_error(self.source_data.crop_images, progress_frame, "Failed to crop images")
            if not self.source_data.stop_event.is_set() and not self.chk_non_linear.GetValue():
                progress_frame.label.SetLabel("Stretching images... ")
                self.handle_load_error(
                    self.source_data.stretch_images, progress_frame, "Failed to stretch images",
                    progress_bar=ProgressBarFactory.create_progress_bar(progress_frame.progress))
            if not self.source_data.stop_event.is_set():
                self.source_data.images_from_buffer()
            else:
                self.source_data.original_frames = None
            self.source_data.stop_event.clear()
            progress_frame.Close()
            wx.CallAfter(self.on_load_finished)
        self.process_thread = threading.Thread(target=load_images_calibrate_and_align)
        self.process_thread.start()

    def on_load_finished(self) -> None:
        """
        Event handler for the images loaded event.
        Displays the selected (first) image on image panel.
        Enables elements responsible for the next step: searching for moving objects and annotation.

        Returns:
            None
        """
        self.source_data.stop_event.clear()
        if self.source_data.images is not None and len(self.source_data.images) > 0:
            short_file_paths = self._gen_short_file_names([item.file_name for item in self.source_data.headers])
            self.checkbox_list.SetObjects([
                MyDataObject(
                    fp, header.timestamp, header.exposure, checked=True
                ) for fp, header in zip(short_file_paths, self.source_data.headers)
            ])
            objects = self.checkbox_list.GetObjects()
            for obj in objects:
                self.checkbox_list.SetCheckState(obj, True)
            self.checkbox_list.SelectObject(objects[0])
            self.checkbox_list.RefreshObjects(objects)
            img_to_draw = self.source_data.images[0]
            img_to_draw = (img_to_draw * 255).astype('uint8')
            self.draw_panel.image_array = img_to_draw
            self.draw_panel.Refresh(eraseBackground=False)
            self.results_label.Enable(True)
            self.magnitude_label.Enable(True)
            self.magnitude_input.Enable(True)
            self.results_path_picker.Enable(True)
            self.btn_process.Enable(True)

    def on_item_selected(self, event: Event) -> None:
        """
        Event handler for when an item is selected.
        Renders the selected item's image on the draw panel.
        """
        if self.source_data.images is not None and len(self.source_data.images) > 0:
            obj = self.checkbox_list.GetSelectedObject()
            file_paths = [item.file_name for item in self.source_data.headers]
            for num, item in enumerate(file_paths):
                if item.endswith(obj.file_path):
                    image_idx = num
                    break
            else:
                raise ValueError(f"{'obj.file_path'} is not in file list")
            img_to_draw = self.source_data.images[image_idx]
            img_to_draw = (img_to_draw * 255).astype('uint8')
            self.draw_panel.image_array = img_to_draw
            self.draw_panel.Refresh(eraseBackground=False)

    def on_process(self, event: Event) -> None:
        """
        Event handler for the 'Process' button click.
        Initiates the process of finding moving objects by AI model.

        Args:
            event (Event): The event object that triggered this function.

        Returns:
            None
        """
        output_folder = self.results_path_picker.GetPath()
        objects = self.checkbox_list.GetObjects()
        use_img_mask = []
        for obj in objects:
            checked = self.checkbox_list.GetCheckState(obj)
            if checked:
                use_img_mask.append(True)
            else:
                use_img_mask.append(False)
        self.source_data.usage_map = np.array(use_img_mask, dtype=bool)

        def find_asteroids():
            """
            Predict asteroids in the given source data using AI model.
            The function to be run in processing thread.
            """
            progress_frame = ProgressFrame(self, "Finding moving objects", stop_event=self.source_data.stop_event)
            progress_frame.progress.SetValue(0)
            progress_frame.label.SetLabel("Finding moving objects...")
            progress_frame.Show()
            results = predict_asteroids(self.source_data, progress_bar=ProgressBarFactory.create_progress_bar(
                progress_frame.progress))
            if not self.source_data.stop_event.is_set():
                progress_frame.label.SetLabel("Saving results...")
                image_to_annotate = save_results(
                    source_data=self.source_data, results=results, output_folder=output_folder)
            if not self.source_data.stop_event.is_set():
                progress_frame.label.SetLabel("Annotating results...")
                magnitude_limit = float(self.magnitude_input.GetValue())
                annotate_results(self.source_data, image_to_annotate, output_folder, magnitude_limit=magnitude_limit)
            self.source_data.stop_event.clear()
            progress_frame.Close()

        self.process_thread = threading.Thread(target=find_asteroids)
        self.process_thread.start()

    def on_process_finished(self) -> None:
        """
        Event handler for the process finished event.
        Closes the progress frame and resets the usage map in source data.
        """
        self.source_data.usage_map = None

    def on_start_again(self, event: Event) -> None:
        """
        Event handler for the 'Start again' button click.
        Resets the source data and sets the startup states.

        Args:
            event (Event): The event object that triggered this function.

        Returns:
            None
        """
        self.source_data = None
        self.set_startup_states()


def start_ui():
    """
    Initializes the user interface and runs the main application loop.
    """
    app = wx.App(False)
    frame = MyFrame(None, wx.ID_ANY, "CelestialSurveyor", style=wx.DEFAULT_FRAME_STYLE)
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    start_ui()
