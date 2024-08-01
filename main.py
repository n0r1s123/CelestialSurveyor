
if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    import argparse
    import os
    import sys

    from backend.find_asteroids import predict_asteroids, save_results, annotate_results
    from backend.progress_bar import ProgressBarCli
    from backend.source_data_v2 import SourceDataV2
    from logger.logger import get_logger
    from user_interface.main_window import start_ui

    logger = get_logger()

    version = "0.3.2"
    arg_parser = argparse.ArgumentParser(
        prog='CelestialSurveyor',
        description='It\'s is designed to analyze astronomical images with the primary goal of identifying and '
                    'locating asteroids and comets within the vastness of the cosmic terrain')

    arg_parser.add_argument('-c', '--cli_mode', dest='cli_mode', action="store_true", default=False,
                            help='Run app in command line mode')
    arg_parser.add_argument('-s', '--source_folder', dest='source_folder', type=str, required=False,
                            help='Path to the folder with xisf or fit or fits files to be analyzed')
    arg_parser.add_argument('-o', '--output_folder', dest='output_folder', type=str, required=False,
                            help='Path to the folder where results will be stored')
    arg_parser.add_argument('--dark_folder', dest='dark_folder', type=str, required=False, default="",
                            help='Path to the folder with Dark frames (Optional)')
    arg_parser.add_argument('--flat_folder', dest='flat_folder', type=str, required=False, default="",
                            help='Path to the folder with Flat frames (Optional). Work only if DarkFlat '
                                 'folder is provided')
    arg_parser.add_argument('--dark_flat_folder', dest='dark_flat_folder', type=str, required=False,
                            default="",
                            help='Path to the folder with DarkFlat frames (Optional). Work only if DarkFlat '
                                 'folder is provided')
    arg_parser.add_argument('-m', '--model_path', dest='model_path',
                            type=str, default="",
                            required=False,
                            help='Path to the AI model file')
    arg_parser.add_argument('-n', '--non_linear', dest='non_linear', action="store_true", required=False,
                            help='Provide this key if the images are not in linear state')
    arg_parser.add_argument('-d', '--debayer', dest='debayer', action="store_true", required=False,
                            default=False, help='Debayer color FIT images if required')
    arg_parser.add_argument('-a', '--align', dest='align_images', action="store_true", required=False,
                            default=False, help='Do image alignment when loading')
    arg_parser.add_argument('-v', '--version', dest='version', action="store_true", required=False,
                            help='Display version of this app.')
    arg_parser.add_argument('-l', '--magnitude_limit', dest='magnitude_limit', type=float, required=False,
                            default='18.0')
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
        if provided_args.magnitude_limit >= 25.0:
            logger.log.error(
                f"Provided magnitude limit {provided_args.magnitude_limit} is too high. Maximum value is 25.0")
            sys.exit(0)
        if provided_args.model_path and not os.path.exists(provided_args.model_path):
            logger.log.error(f"Provided model path {provided_args.model_path} does not exist")
            sys.exit(0)
        if provided_args.flat_folder and not os.path.exists(provided_args.flat_folder):
            logger.log.error(f"Provided flat folder {provided_args.flat_folder} does not exist")
            sys.exit(0)
        if provided_args.dark_folder and not os.path.exists(provided_args.dark_folder):
            logger.log.error(f"Provided dark folder {provided_args.dark_folder} does not exist")
            sys.exit(0)
        if provided_args.dark_flat_folder and not os.path.exists(provided_args.dark_flat_folder):
            logger.log.error(f"Provided dark flat folder {provided_args.dark_flat_folder} does not exist")
            sys.exit(0)

        source_data = SourceDataV2(provided_args.debayer)
        img_paths = source_data.make_file_paths(provided_args.source_folder)
        source_data.extend_headers(file_list=img_paths)
        source_data.load_images(progress_bar=ProgressBarCli())
        dark_files = source_data.make_file_paths(provided_args.dark_folder) if provided_args.dark_folder else None
        flat_files = source_data.make_file_paths(provided_args.flat_folder) if provided_args.flat_folder else None
        dark_flat_files = source_data.make_file_paths(
            provided_args.dark_flat_folder) if provided_args.dark_flat_folder else None

        source_data.calibrate_images(
            dark_files=dark_files,
            flat_files=flat_files,
            dark_flat_files=dark_flat_files,
            progress_bar=ProgressBarCli())
        if provided_args.align_images:
            source_data.plate_solve_all(progress_bar=ProgressBarCli())
            source_data.align_images_wcs(progress_bar=ProgressBarCli())
        source_data.crop_images()
        if not provided_args.non_linear:
            source_data.stretch_images(progress_bar=ProgressBarCli())
        source_data.images_from_buffer()

        results = predict_asteroids(source_data, model_path=provided_args.model_path, progress_bar=ProgressBarCli())
        image_to_annotate = save_results(source_data=source_data, results=results,
                                         output_folder=provided_args.output_folder)

        magnitude_limit = float(provided_args.magnitude_limit)
        annotate_results(source_data, image_to_annotate, provided_args.output_folder, magnitude_limit=magnitude_limit)
        logger.log.info(f"Results are saved in {provided_args.output_folder}")
