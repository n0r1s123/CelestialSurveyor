import cv2
import datetime
import h5py
import numpy as np
import os
import tensorflow as tf

from cryptography.fernet import Fernet
from io import BytesIO
from PIL import Image
from typing import Optional

from backend.consuming_functions.measure_execution_time import measure_execution_time
from backend.progress_bar import AbstractProgressBar
from backend.source_data_v2 import SourceDataV2, CHUNK_SIZE
from logger.logger import get_logger


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = get_logger()


@measure_execution_time
@tf.autograph.experimental.do_not_convert
def predict_asteroids(source_data: SourceDataV2, progress_bar: Optional[AbstractProgressBar] = None,
                      model_path: Optional[str] = None) -> list[tuple[int, int, float]]:
    """
    Predict asteroids in the given source data using AI model.

    Args:
        source_data (SourceDataV2): The source data containing calibrated and aligned monochromatic images.
        progress_bar (Optional[AbstractProgressBar], optional): Progress bar for tracking prediction progress.
            Defaults to None.
        model_path (Optional[str], optional): Path to the AI model for prediction. If the path is not provided, the
            model with the highest version will be used. Defaults to None.

    Returns:
        List[Tuple[int, int, float]]: A list of tuples containing the coordinates and confidence level of predicted
            asteroids.
    """
    logger.log.info("Finding moving objects...")
    model_path = get_model_path() if model_path is None else model_path
    logger.log.info(f"Loading model: {model_path}")
    model = decrypt_model(model_path)
    batch_size = 10
    logger.log.debug(f"Batch size: {batch_size}")
    chunk_generator = source_data.generate_image_chunks()
    batch_generator = source_data.generate_batch(chunk_generator, batch_size=batch_size)
    ys, xs = source_data.get_number_of_chunks()
    progress_bar_len = len(ys) * len(xs)
    progress_bar_len = progress_bar_len // batch_size + 1 if progress_bar_len % batch_size != 0 else 0
    if progress_bar:
        progress_bar.set_total(progress_bar_len)
    objects_coords = []
    for coords, batch in batch_generator:
        if source_data.stop_event.is_set():
            break
        results = model.predict(batch, verbose=0)
        for res, (y, x) in zip(results, coords):
            if res > 0.8:
                objects_coords.append((y, x, res))
        if progress_bar:
            progress_bar.update()
    progress_bar.complete()
    return objects_coords


def save_results(source_data: SourceDataV2, results, output_folder) -> np.ndarray:
    """
    This function saves the results of the object recognition process. It marks the areas where the model located
    probable asteroids and creates the GIFs of these areas.

    Args:
        source_data (SourceDataV2): The source data object containing image data.
        results: The results of the object recognition process.
        output_folder (str): The folder where the results will be saved.
    """
    logger.log.info("Saving results...")
    max_image = np.copy(source_data.max_image) * 255.
    max_image = cv2.cvtColor(max_image, cv2.COLOR_GRAY2BGR)
    gif_size = 5
    processed = []
    for coord_num, (y, x, probability) in enumerate(results):
        probability = probability[0]
        color = (0, 255, 255) if probability <= 0.9 else (0, 255, 0)
        max_image = cv2.rectangle(max_image, (x, y), (x+CHUNK_SIZE, y+CHUNK_SIZE), color, 2)
        max_image = cv2.putText(max_image, "{:.2f}".format(probability), org=(x, y - 10),
                            fontFace=1, fontScale=1, color=(0, 0, 255), thickness=0)
        for y_pr, x_pr in processed:
            if x_pr - (gif_size // 2) * 64 <= x <= x_pr + (gif_size // 2) * 64 and \
                    y_pr - (gif_size // 2) * 64 <= y <= y_pr + (gif_size // 2) * 64:
                break
        else:
            processed.append((y, x))
            y_new, x_new, size = get_big_rectangle_coords(y, x, max_image.shape, gif_size)
            max_image = cv2.rectangle(max_image, (x_new, y_new), (x_new + size, y_new + size), (0, 0, 255), 4)
            max_image = cv2.putText(max_image, str(len(processed)), org=(x_new + 20, y_new + 60),
                            fontFace=1, fontScale=3, color=(0, 0, 255), thickness=2)
            frames = source_data.crop_image(
                source_data.images,
                (y_new, y_new + size),
                (x_new, x_new + size))
            frames = frames * 255
            new_shape = list(frames.shape)
            new_shape[1] += 20
            new_frames = np.zeros(new_shape)
            new_frames[:, :-20, :] = frames
            used_timestamps = [item.timestamp for item in source_data.headers]
            for frame, original_ts in zip(new_frames, used_timestamps):
                cv2.putText(frame, text=original_ts.strftime("%d/%m/%Y %H:%M:%S %Z"), org=(70, 64 * gif_size + 16),
                            fontFace=1, fontScale=1, color=(255, 255, 255), thickness=0)
            new_frames = [Image.fromarray(frame.reshape(frame.shape[0], frame.shape[1])).convert('L').convert('P') for frame in new_frames]
            new_frames[0].save(
                os.path.join(output_folder, f"{len(processed)}.gif"),
                save_all=True,
                append_images=new_frames[1:],
                duration=200,
                loop=0)
    cv2.imwrite(os.path.join(output_folder, "results.png"), max_image)
    return max_image


def annotate_results(source_data: SourceDataV2, img_to_be_annotated: np.ndarray, output_folder: str,
                     magnitude_limit: float) -> None:
    """
    Annotates the results on the input image. If there are known asteroids or comets within Field of View - they will
    be marked. Annotation will be done for the first timestamp of each imaging session.

    Args:
        source_data (SourceDataV2): The source data object containing image data.
        img_to_be_annotated (np.ndarray): The image to be annotated.
        output_folder (str): The folder where the annotated results will be saved.
        magnitude_limit (float): The magnitude limit for known asteroids.
    """
    logger.log.info("Annotating results...")
    start_session_frame_nums = [0]

    start_ts = source_data.headers[0].timestamp
    for num, header in enumerate(source_data.headers[1:], start=1):
        if header.timestamp - start_ts > datetime.timedelta(hours=12):
            start_session_frame_nums.append(num)
            start_ts = header.timestamp
    for num, start_frame_num in enumerate(start_session_frame_nums, start=1):
        logger.log.info(f"Fetching known objects for session number {num}")
        for obj_type in source_data.fetch_known_asteroids_for_image(start_frame_num, magnitude_limit=magnitude_limit):
            for item in obj_type:
                target_x, target_y = item.pixel_coordinates
                target_x = round(float(target_x))
                target_y = round(float(target_y))
                x = (target_x, target_x)
                if target_y < 50:
                    y = (target_y + 4, target_y + 14)
                else:
                    y = (target_y - 4, target_y - 14)
                img_to_be_annotated =cv2.line(
                    img_to_be_annotated, (x[0], y[0]), (x[1], y[1]), (0, 165, 255), 2)

                if target_y < 50:
                    text_y = target_y + 20 + 20
                else:
                    text_y = target_y - 20

                if target_x > source_data.shape[1] - 300:
                    text_x = target_x - 300
                else:
                    text_x = target_x
                img_to_be_annotated = cv2.putText(img_to_be_annotated, f"{item.name}: {item.magnitude}", org=(text_x, text_y),
                                        fontFace=0, fontScale=1, color=(0, 165, 255), thickness=2)

    cv2.imwrite(os.path.join(output_folder, "results_annotated.png"), img_to_be_annotated)


# Decrypt the model weights
def decrypt_model(encrypted_model_path: str,
                  key: Optional[bytes] = b'J17tdv3zz2nemLNwd17DV33-sQbo52vFzl2EOYgtScw=') -> tf.keras.Model:
    """
    Decrypts the model weights using the provided key.
    Preliminary version of model encryption. It was done when I was thinking to make this project open source or not.

    Args:
        encrypted_model_path (str): The path to the encrypted model file.
        key (bytes): The key used for decryption. Defaults to a preset key.

    Returns:
        tf.keras.Model: The decrypted model.
    """
    # Read the encrypted weights from the file
    with open(encrypted_model_path, "rb") as file:
        encrypted_model_data = file.read()

    # Use the provided key to create a cipher
    cipher = Fernet(key)

    # Decrypt the entire model
    decrypted_model_data = cipher.decrypt(encrypted_model_data)

    # Load the decrypted model directly into memory
    decrypted_model_data = BytesIO(decrypted_model_data)
    h = h5py.File(decrypted_model_data, 'r')
    loaded_model = tf.keras.models.load_model(h)

    return loaded_model


def get_model_path() -> str:
    """
    Get the path to the latest model file.

    Returns:
        str: The path to the latest model file.
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    file_list = []
    if os.path.exists(root_dir):
        file_list = os.listdir(root_dir)
    models = [item for item in file_list if item.startswith('model') and item.endswith('bin')]
    model_nums = []
    for model in models:
        name, _ = model.split('.')
        num = name[5:]
        model_nums.append(int(num))
    if model_nums:
        model_num = max(model_nums)
        model_path = root_dir
    else:
        secondary_dir = os.path.split(root_dir)[0]
        file_list = os.listdir(secondary_dir)
        models = [item for item in file_list if item.startswith('model') and item.endswith('bin')]
        model_nums = []
        for model in models:
            name, _ = model.split('.')
            num = name[5:]
            model_nums.append(int(num))
        if model_nums:
            model_num = max(model_nums)
            model_path = secondary_dir
        else:
            raise Exception("AI model was not found.")
    model_path = os.path.join(model_path, f"model{model_num}.bin")
    return model_path


def get_big_rectangle_coords(y: int, x: int, image_shape: tuple, gif_size: int) -> tuple[int, int, int]:
    """
    Calculate the coordinates of a large rectangle based on the center coordinates and image properties.
    Large rectangle is used for the GIF animation.

    Args:
        y (int): The y-coordinate of the center point.
        x (int): The x-coordinate of the center point.
        image_shape (Tuple[int, int, int]): The shape of the image (height, width, channels).
        gif_size (int): The size of the GIF.

    Returns:
        Tuple[int, int, int]: The coordinates of the top-left corner of the rectangle (box_y, box_x)
        and the size of the rectangle.
    """
    size = CHUNK_SIZE
    box_x = 0 if x - size * (gif_size // 2) < 0 else x - size * (gif_size // 2)
    box_y = 0 if y - size * (gif_size // 2) < 0 else y - size * (gif_size // 2)
    image_size_y, image_size_x = image_shape[:2]
    box_x = image_size_x - size * gif_size if x + size * (gif_size // 2 + 1) > image_size_x else box_x
    box_y = image_size_y - size * gif_size if y + size * (gif_size // 2 + 1) > image_size_y else box_y
    return box_y, box_x, size * gif_size
