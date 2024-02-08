# import os
import wx
# import numpy as np
from ObjectListView import ObjectListView, ColumnDefn
from PIL import Image
import threading
from backend.find_asteroids import find_asteroids

from backend.source_data import SourceData
import wx.lib.scrolledpanel as scrolled
from logger.logger import get_logger
from backend.progress_bar import ProgressBarFactory

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

        self.Refresh()

    def convert_array_to_bitmap(self, array, display_width=None, display_height=None):
        # Create a PIL Image from the NumPy array
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


class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)

        self.source_data = None

        # Create the main panel
        self.panel = wx.Panel(self)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)

        # Create a horizontal box sizer to organize the elements
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        # Create the controls area (1/6 of window width)
        controls_sizer = wx.BoxSizer(wx.VERTICAL)

        controls_label = wx.StaticText(self.panel, label="Controls:")
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
        choices = [str(i) for i in range(1, 4)]
        self.chk_to_second_align = wx.CheckBox(self.panel, label="Secondary alignment")
        self.chk_to_second_align.SetValue(False)
        label1 = wx.StaticText(self.panel, label="Select number of X splits for secondary alignment")
        self.choice_x_splits = wx.Choice(self.panel, choices=choices)
        self.choice_x_splits.SetStringSelection("3")
        label2 = wx.StaticText(self.panel, label="Select number of Y splits for secondary alignment")
        self.choice_y_splits = wx.Choice(self.panel, choices=choices)
        self.choice_y_splits.SetStringSelection("3")
        label3 = wx.StaticText(self.panel, label="Select folder to store results")
        self.results_path_picker = wx.DirPickerCtrl(self.panel, style=wx.DIRP_USE_TEXTCTRL)
        self.progress_bar = wx.Gauge(self.panel, range=10)
        self.process_progress_bar = wx.Gauge(self.panel, range=10)

        self.btn_process = wx.Button(self.panel, label="Process")
        self.Bind(wx.EVT_BUTTON, self.on_process, self.btn_process)
        self.btn_start_again = wx.Button(self.panel, label="Start again")
        self.Bind(wx.EVT_BUTTON, self.on_start_again, self.btn_start_again)

        controls_sizer.Add(controls_label, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(dark_label, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.dark_path_picker, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.btn_add_files, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.to_debayer, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.chk_to_align, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.chk_non_linear, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.btn_load_files, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.progress_bar, 0, wx.EXPAND | wx.ALL, 5)

        controls_sizer.Add(self.chk_to_second_align, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(label1, 0, wx.ALL, 5)
        controls_sizer.Add(self.choice_x_splits, 0, wx.ALL, 5)
        controls_sizer.Add(label2, 0, wx.ALL, 5)
        controls_sizer.Add(self.choice_y_splits, 0, wx.ALL, 5)
        controls_sizer.Add(label3, 0, wx.ALL, 5)
        controls_sizer.Add(self.results_path_picker, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.btn_process, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.process_progress_bar, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(self.btn_start_again, 0, wx.EXPAND | wx.ALL, 5)

        hbox.Add(controls_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Create the checkbox list area (2/6 of window width) using ObjectListView
        checkbox_label = wx.StaticText(self.panel, label="Checkbox List:")
        self.checkbox_list = ObjectListView(self.panel, style=wx.LC_REPORT | wx.SUNKEN_BORDER)
        self.checkbox_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_item_selected)
        columns = [
            ColumnDefn("Item", "left", 150, "file_path"),
            ColumnDefn("Timestamp", "left", 100, "timestamp"),
            ColumnDefn("Exposure", "left", 100, "exposure"),
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
        self.SetTitle("wxPython Example")
        self.Centre()
        self.Maximize(True)

        # Bind events
        self.Bind(wx.EVT_MENU, self.OnExit, id=wx.ID_EXIT)
        logger.log.info("App initialized")

    def set_startup_states(self):
        self.chk_to_align.Enable(False)
        self.to_debayer.Enable(False)
        self.chk_non_linear.Enable(False)
        self.btn_load_files.Enable(False)
        self.chk_to_second_align.Enable(False)
        self.choice_x_splits.Enable(False)
        self.choice_y_splits.Enable(False)
        self.results_path_picker.Enable(False)
        self.btn_process.Enable(False)
        self.progress_bar.SetValue(0)
        self.process_progress_bar.SetValue(0)
        self.checkbox_list.SetObjects([])
        self.checkbox_list.Refresh()
        self.results_path_picker.SetPath('')

    def OnExit(self, event):
        self.Close()

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
                self.source_data.file_list.extend(paths)
            else:
                self.source_data = SourceData(paths)
            self.source_data.load_headers_and_sort()
            self.checkbox_list.SetObjects([])
            self.checkbox_list.Refresh()
            self.checkbox_list.AddObjects([
                MyDataObject(
                    fp, timestamp, exposure, checked=True
                ) for fp, timestamp, exposure in self.source_data.timestamped_file_list
            ])
            objects = self.checkbox_list.GetObjects()
            for obj in objects:
                self.checkbox_list.SetCheckState(obj, True)
            self.checkbox_list.RefreshObjects(objects)

        dialog.Destroy()
        self.progress_bar.SetValue(0)
        self.process_progress_bar.SetValue(0)
        logger.log.debug(f"Added the following files: {paths}")
        if paths:
            self.btn_load_files.Enable(True)
            self.chk_to_align.Enable(True)
            self.to_debayer.Enable(True)
            self.chk_non_linear.Enable(True)


    def on_load_files(self, event):
        logger.log.debug("Loading files is initialized by user")
        new_timestamped_file_list = []
        objects = self.checkbox_list.GetObjects()
        logger.log.debug(f"Excluding unchecked files from file list: {[item[0] for item in self.source_data.timestamped_file_list]}")
        for file_list_obj, obj in zip(self.source_data.timestamped_file_list, objects):
            checked = self.checkbox_list.GetCheckState(obj)
            if checked:
                new_timestamped_file_list.append(file_list_obj)
            else:
                logger.log.debug(f"Excluding {file_list_obj[0]}")
        self.source_data.timestamped_file_list = new_timestamped_file_list
        self.checkbox_list.SetObjects([
            MyDataObject(
                fp, timestamp, exposure, checked=True
            ) for fp, timestamp, exposure in self.source_data.timestamped_file_list
        ])
        objects = self.checkbox_list.GetObjects()
        for obj in objects:
            self.checkbox_list.SetCheckState(obj, True)
        self.checkbox_list.RefreshObjects(objects)

        to_align = self.chk_to_align.GetValue()
        self.source_data.to_align = to_align

        logger.log.info("Starting image loading...")
        dark_path = self.dark_path_picker.GetPath()
        dark_path = dark_path if dark_path else None
        logger.log.error(dark_path)
        self.process_thread = threading.Thread(target=self.source_data.load_images, kwargs={
            "progress_bar": ProgressBarFactory.create_progress_bar(self.progress_bar),
            "frame": self,
            "to_debayer": self.to_debayer.GetValue(),
            "dark_folder": dark_path

        })
        self.process_thread.start()

    def on_load_finished(self):
        self.checkbox_list.SetObjects([
            MyDataObject(
                fp, timestamp, exposure, checked=True
            ) for fp, timestamp, exposure in self.source_data.timestamped_file_list
        ])
        objects = self.checkbox_list.GetObjects()
        for obj in objects:
            self.checkbox_list.SetCheckState(obj, True)
        self.checkbox_list.SelectObject(objects[0])
        self.checkbox_list.RefreshObjects(objects)
        img_to_draw = (self.source_data.images[0] * 255).astype('uint8')
        self.draw_panel.image_array = img_to_draw
        self.draw_panel.Refresh()
        self.chk_to_second_align.Enable(True)
        self.choice_x_splits.Enable(True)
        self.choice_y_splits.Enable(True)
        self.results_path_picker.Enable(True)
        self.btn_process.Enable(True)

    def on_item_selected(self, event):
        if self.source_data.images is not None and len(self.source_data.images) > 0:
            obj = self.checkbox_list.GetSelectedObject()
            file_paths = [item[0] for item in self.source_data.timestamped_file_list]
            image_idx = file_paths.index(obj.file_path)
            img_to_draw = (self.source_data.images[image_idx] * 255).astype('uint8')
            self.draw_panel.image_array = img_to_draw
            self.draw_panel.Refresh()

    def on_process(self, event):
        output_folder = self.results_path_picker.GetPath()
        objects = self.checkbox_list.GetObjects()
        use_img_mask = []
        for obj in objects:
            checked = self.checkbox_list.GetCheckState(obj)
            if checked:
                use_img_mask.append(True)
            else:
                use_img_mask.append(False)

        y_splits = int(self.choice_y_splits.GetString(self.choice_y_splits.GetCurrentSelection()))
        x_splits = int(self.choice_x_splits.GetString(self.choice_x_splits.GetCurrentSelection()))

        self.process_thread = threading.Thread(target=find_asteroids, kwargs={
            "source_data": self.source_data,
            "use_img_mask": use_img_mask,
            "output_folder": output_folder,
            "y_splits": y_splits,
            "x_splits": x_splits,
            "secondary_alignment": self.chk_to_second_align.GetValue(),
            "progress_bar": ProgressBarFactory.create_progress_bar(self.process_progress_bar),

        })
        self.process_thread.start()

    def on_start_again(self, event):
        self.source_data = None
        self.set_startup_states()


def start_ui():
    app = wx.App(False)
    frame = MyFrame(None, wx.ID_ANY, "wxPython Window", style=wx.DEFAULT_FRAME_STYLE)
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    start_ui()
