import sys

import argparse
import os
import tqdm

from backend.find_asteroids import find_asteroids
from backend.source_data import SourceData, get_file_paths
from user_interface.main_window import start_ui
from logger.logger import get_logger
from backend.progress_bar import ProgressBarFactory
import multiprocessing


logger = get_logger()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    version = "0.2.0"
    arg_parser = argparse.ArgumentParser(
        prog='CelestialSurveyor',
        description='It\'s is designed to analyze astronomical images with the primary goal of identifying and '
                    'locating asteroids and comets within the vastness of the cosmic terrain')

    arg_parser.add_argument('-c', '--cli_mode', dest='cli_mode', action="store_true",
                            help='Run app in command line mode')
    arg_parser.add_argument('-s', '--source_folder', dest='source_folder', type=str, required=False,
                            help='Path to the folder with xisf or fit or fits files to be analyzed')
    arg_parser.add_argument('-o', '--output_folder', dest='output_folder', type=str, required=False,
                            help='Path to the folder where results will be stored')
    arg_parser.add_argument('--dark_folder', dest='dark_folder', type=str, required=False, default=None,
                            help='Path to the folder with Dark frames (Optional)')
    arg_parser.add_argument('--flat_folder', dest='flat_folder', type=str, required=False, default=None,
                            help='Path to the folder with Flat frames (Optional). Work only if DarkFlat '
                                 'folder is provided')
    arg_parser.add_argument('--dark_flat_folder', dest='dark_flat_folder', type=str, required=False, default=None,
                            help='Path to the folder with DarkFlat frames (Optional). Work only if DarkFlat '
                                 'folder is provided')
    arg_parser.add_argument('-m', '--model_path', dest='model_path', type=str, default="default", required=False,
                            help='Path to the AI model file')
    arg_parser.add_argument('-n', '--non_linear', dest='non_linear', action="store_true", required=False,
                            help='Provide this key if the images are not in linear state')
    arg_parser.add_argument('-d', '--debayer', dest='debayer', action="store_true", required=False, default=False,
                            help='Debayer color FIT images if required')
    arg_parser.add_argument('-a', '--initial_alignment', dest='initial_alignment', action="store_true", required=False,
                            default=False, help='Do image alignment when loading')
    arg_parser.add_argument('--not_skip_bad', dest='not_skip_bad', action="store_true", required=False,
                            default=False, help='App will finish execution if there are images failed to '
                                                'be aligned in case if this key is provided. Otherwise they will be '
                                                'skipped')
    arg_parser.add_argument('--secondary_alignment', dest='secondary_alignment', action="store_true", required=False,
                            default=False, help='Do secondary alignment. Helpful when there are rotation between '
                                                'frames. In this case each image will be split by x_splits*y_splits '
                                                'plates and aligned respectively')
    arg_parser.add_argument('--x_splits', dest='x_splits', type=int, default=1, required=False,
                            help='Number of X-splits. Check "--secondary_alignment" key')
    arg_parser.add_argument('--y_splits', dest='y_splits', type=int, default=1, required=False,
                            help='Number of Y-splits. Check "--secondary_alignment" key')
    arg_parser.add_argument('-v', '--version', dest='version', action="store_true", required=False,
                            help='Display version of this app.')
    provided_args = arg_parser.parse_args()
    if provided_args.version:
        print(f"CelestialSurveyor v{version}")
    if not provided_args.cli_mode:
        start_ui()
    else:
        if not os.path.exists(provided_args.source_folder):
            logger.log.error(f"Provided source folder {provided_args.source_folder} does not exist")
            sys.exit(0)
        if not os.path.exists(provided_args.output_folder):
            try:
                os.makedirs(provided_args.output_folder)
            except Exception as e:
                logger.log.error(f"Unable to create output folder {provided_args.output_folder} due to {str(e)}")
                sys.exit(0)
        if provided_args.model_path != "default":
            if not os.path.exists(provided_args.model_path):
                logger.log.error(f"Provided model path {provided_args.model_path} does not exist")
                sys.exit(0)

        progress_bar = tqdm.tqdm()
        source_data = SourceData(
            file_list=get_file_paths(provided_args.source_folder),
            non_linear=provided_args.non_linear,
            to_align=provided_args.initial_alignment,
            to_skip_bad=not provided_args.not_skip_bad,
            num_from_session=15,
            dark_folder=provided_args.dark_folder,
            flat_folder=provided_args.flat_folder,
            dark_flat_folder=provided_args.dark_flat_folder,
            to_debayer=provided_args.debayer,

        )
        source_data.load_headers_and_sort()
        source_data.load_images(
            progress_bar=ProgressBarFactory.create_progress_bar(tqdm.tqdm()),
        )
        find_asteroids(
            source_data=source_data,
            use_img_mask=None,
            output_folder=provided_args.output_folder,
            secondary_alignment=provided_args.secondary_alignment,
            y_splits=provided_args.y_splits,
            x_splits=provided_args.x_splits,
            progress_bar=ProgressBarFactory.create_progress_bar(tqdm.tqdm())
        )
