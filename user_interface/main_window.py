# import os
import os.path
import time

import wx
# import numpy as np
from ObjectListView import ObjectListView, ColumnDefn
from PIL import Image
import threading
from backend.find_asteroids import predict_asteroids, save_results, annotate_results

from backend.source_data_v2 import SourceDataV2
import wx.lib.scrolledpanel as scrolled
from logger.logger import get_logger
from backend.progress_bar import ProgressBarFactory
from threading import Event
import numpy as np
import json

logger = get_logger()


class MyDataObject:
    def __init__(self, item, timestamp, exposure, checked=False):
        self.file_path = item
        self.timestamp = timestamp
        self.exposure = exposure
        self.checked = checked


class ImagePanel(scrolled.ScrolledPanel):
    def __init__(self, parent, image_array=None):
        super().__init__(parent, style=wx.BORDER_SIMPLE)
        self.image_array = image_array
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_scroll)

        self.scale_factor = None
        self.default_scale_factor = 1.0
        self.scroll_x = 0
        self.scroll_y = 0

        self.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)

    def on_paint(self, event):
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


    def on_scroll(self, event):
        rotation = event.GetWheelRotation()
        lines_per_scroll = event.GetLinesPerAction()

        # Calculate the new scale factor based on the mouse wheel rotation
        if rotation > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1

        self.Refresh(eraseBackground=False)

    def convert_array_to_bitmap(self, array, display_width=None, display_height=None):
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


class ProgressFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent=parent, title=title, size=(500, 200))
        self.parent = parent
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

    def on_cancel(self, event):
        self.label.SetLabel("Stopping...")
        if self.parent.source_data.load_processes:
            while any(item.pid is None for item in self.parent.source_data.load_processes):
                time.sleep(0.1)
        self.parent.stop_event.set()
        self.Close()


class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)

        self.source_data: SourceDataV2 = None
        self

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

        self.Bind(wx.EVT_BUTTON, self.on_load_files, self.btn_load_files)
        choices = [str(i) for i in range(1, 11)]
        self.chk_to_second_align = wx.CheckBox(self.panel, label="Secondary alignment")
        self.chk_to_second_align.SetValue(True)
        label1 = wx.StaticText(self.panel, label="Select number of X splits for secondary alignment")
        self.choice_x_splits = wx.Choice(self.panel, choices=choices)
        self.choice_x_splits.SetStringSelection("3")
        label2 = wx.StaticText(self.panel, label="Select number of Y splits for secondary alignment")
        self.choice_y_splits = wx.Choice(self.panel, choices=choices)
        self.choice_y_splits.SetStringSelection("3")
        label3 = wx.StaticText(self.panel, label="Select folder to store results")
        self.results_path_picker = wx.DirPickerCtrl(self.panel, style=wx.DIRP_USE_TEXTCTRL)

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
        controls_sizer.Add(self.results_path_picker, 0, wx.EXPAND | wx.ALL, 5)
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
        self.SetSize((800, 300))
        self.SetTitle("CelestialSurveyor")
        self.Centre()
        self.Maximize(True)

        # Bind events
        self.Bind(wx.EVT_MENU, self.OnExit, id=wx.ID_EXIT)
        logger.log.info("App initialized")

    def __get_previous_settings(self):
        settings_file = "settings.json"
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                return json.load(f)
        else:
            return {}

    def __write_previous_settings(self, settings):
        with open("settings.json", 'w') as f:
            json.dump(settings, f)

    def set_startup_states(self):
        prev_settings = self.__get_previous_settings()
        dark_path = prev_settings.get('dark_path', '')
        self.dark_path_picker.SetPath(dark_path)
        flat_path = prev_settings.get('flat_path', '')
        self.flat_path_picker.SetPath(flat_path)
        dark_flat_path = prev_settings.get('dark_flat_path', '')
        self.dark_flat_path_picker.SetPath(dark_flat_path)
        self.chk_to_align.Enable(False)
        self.to_debayer.Enable(False)
        self.chk_non_linear.Enable(False)
        self.btn_load_files.Enable(False)
        self.results_path_picker.Enable(False)
        self.btn_process.Enable(False)
        self.checkbox_list.SetObjects([])
        self.checkbox_list.Refresh(eraseBackground=False)
        self.results_path_picker.SetPath('')

    def OnExit(self, event):
        self.Close()

    def _gen_short_file_names(self, file_list):
        folders = {os.path.split(fp)[0] for fp in file_list}
        split = os.path.split
        if len(folders) == 1:
            short_file_names = [split(fp)[1] for fp in file_list]
        else:
            short_file_names = [os.path.join(split(split(fp)[0])[1], split(fp)[1]) for fp in file_list]
        return short_file_names


    def on_add_files(self, event):
        wildcard = "Fits files (*.fits)|*.fits;*.FITS;*.fit;*.FIT|XISF files (*.xisf)|*.xisf;*.XISF|All files (*.*)|*.*"

        dialog = wx.FileDialog(self, message="Choose files to load",
                               wildcard=wildcard,
                               style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)

        paths = []
        if dialog.ShowModal() == wx.ID_OK:
            paths = dialog.GetPaths()
            # TODO: Populate other parameters
            if self.source_data:
                self.source_data.extend_headers(paths)
            else:
                self.source_data: SourceDataV2 = SourceDataV2(paths)
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
        dialog.Destroy()
        logger.log.debug(f"Added the following files: {paths}")
        if paths:
            self.btn_load_files.Enable(True)
            self.chk_to_align.Enable(True)
            self.to_debayer.Enable(True)
            self.chk_non_linear.Enable(True)

    def on_load_files(self, event):
        self.progress_frame = ProgressFrame(self, "Loading progress")
        self.progress_frame.progress.SetValue(0)

        # self.progress_bar.SetValue(0)
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

        self.to_align = self.chk_to_align.GetValue()

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
        self.dark_folder = dark_path
        self.flat_folder = flat_path
        self.dark_flat_folder = dark_flat_path
        self.source_data.to_debayer = self.to_debayer.GetValue()
        self.source_data.linear = not self.chk_non_linear.GetValue()
        self.stop_event = Event()

        def load_images_calibrate_and_align():
            self.progress_frame.label.SetLabel("Loading images...")
            self.source_data.load_images(progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))
            self.progress_frame.label.SetLabel("Loading dark...")
            master_dark = self.source_data.make_master_dark(
                self.source_data.make_file_paths(dark_path),
                progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))
            self.progress_frame.label.SetLabel("Loading dark flat...")
            master_dark_flat = self.source_data.make_master_dark(
                self.source_data.make_file_paths(dark_flat_path),
                progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))
            self.progress_frame.label.SetLabel("Loading flats...")
            flats = self.source_data.load_flats(
                self.source_data.make_file_paths(flat_path),
                progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))
            self.source_data.original_frames -= master_dark
            for flat in flats:
                flat -= master_dark_flat
            master_flat = np.average(flats, axis=0)
            self.source_data.original_frames /= master_flat
            self.progress_frame.label.SetLabel("Plate solving...")
            self.source_data.plate_solve_all(progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))
            if self.to_align:
                self.progress_frame.label.SetLabel("Aligning images...")
                self.source_data.align_images_wcs(progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))
            self.progress_frame.label.SetLabel("Cropping images...")
            self.source_data.crop_images()
            # if self.to_align:
            #     self.progress_frame.label.SetLabel("Secondary aligning images... ")
            #     self.source_data.secondary_align_images(progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))

            if not self.chk_non_linear.GetValue():
                self.progress_frame.label.SetLabel("Stretching images... ")
                self.source_data.stretch_images(progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))
            self.source_data.images_from_buffer()

            self.progress_frame.Close()
            wx.CallAfter(self.on_load_finished)
        self.process_thread = threading.Thread(target=load_images_calibrate_and_align)
        self.progress_frame.Show()
        self.process_thread.start()

    def on_load_finished(self):
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
        self.results_path_picker.Enable(True)
        self.btn_process.Enable(True)

    def on_item_selected(self, event):
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

    def on_process(self, event):
        self.progress_frame = ProgressFrame(self, "Finding moving objects")
        self.progress_frame.progress.SetValue(0)
        output_folder = self.results_path_picker.GetPath()
        objects = self.checkbox_list.GetObjects()
        use_img_mask = []
        for obj in objects:
            checked = self.checkbox_list.GetCheckState(obj)
            if checked:
                use_img_mask.append(True)
            else:
                use_img_mask.append(False)
        # self.stop_event = Event()

        def find_asteroids():
            self.progress_frame.label.SetLabel("Finding moving objects...")
            results = predict_asteroids(self.source_data, use_img_mask, progress_bar=ProgressBarFactory.create_progress_bar(self.progress_frame.progress))
            self.progress_frame.label.SetLabel("Saving results...")
            plt = save_results(source_data=self.source_data, results=results, output_folder=output_folder)
            self.progress_frame.label.SetLabel("Annotating results...")
            annotate_results(self.source_data, plt, output_folder)
            self.progress_frame.Close()

        self.progress_frame.Show()
        self.process_thread = threading.Thread(target=find_asteroids)
        self.process_thread.start()

    def on_process_finished(self):
        self.progress_frame.Close()


    def on_start_again(self, event):
        self.source_data = None
        self.set_startup_states()


def start_ui():
    app = wx.App(False)
    frame = MyFrame(None, wx.ID_ANY, "CelestialSurveyor", style=wx.DEFAULT_FRAME_STYLE)
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    start_ui()
