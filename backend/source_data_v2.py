import os
from backend.consuming_functions import load_headers, load_images


class SourceDataV2:
    """
    Represents a source data object with multiple frames.
    """
    def __init__(self, file_list: list[str]) -> None:
        self.headers = load_headers(file_list)
        self.headers.sort(key=lambda header: header.timestamp)
        self._original_frames = None

    @property
    def num_frames(self):
        return len(self.headers)

    @property
    def original_shape(self):
        return self._original_frames.shape if self._original_frames is not None else None

    def load_images(self):
        file_list = [header.file_name for header in self.headers]
        self._original_frames = load_images(file_list)


if __name__ == '__main__':
    folder = "D:\\git\\dataset\\Seahorse\\cropped"
    file_list = os.listdir(folder)
    file_list = [os.path.join(folder, item) for item in file_list]
    source_data = SourceDataV2(file_list)
    source_data.load_images()
