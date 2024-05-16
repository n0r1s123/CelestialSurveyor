import datetime
import os
import numpy as np
import wx
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle
import cv2

from cryptography.fernet import Fernet
from io import BytesIO
import h5py
import tensorflow as tf
from PIL import Image

# from .source_data import SourceData, stretch_image
from backend.source_data_v2 import SourceDataV2, CHUNK_SIZE
from .progress_bar import AbstractProgressBar

from logger.logger import get_logger
from threading import Event


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = get_logger()


def predict_asteroids(source_data: SourceDataV2, use_img_mask,
                      progress_bar: Optional[AbstractProgressBar] = None, model_path="default"):
    if use_img_mask is None:
        use_img_mask = [True for _ in range(len(source_data.images))]

    model_path = get_model_path() if model_path == "default" else model_path
    logger.log.info(f"Loading model: {model_path}")
    model = decrypt_model(model_path)
    batch_size = 10
    chunk_generator = source_data.generate_image_chunks()
    batch_generator = source_data.generate_batch(chunk_generator, batch_size=batch_size, usage_mask=use_img_mask)
    ys, xs = source_data.get_number_of_chunks()
    progress_bar_len = len(ys) * len(xs)

    if progress_bar:
        progress_bar.set_total(progress_bar_len)
    objects_coords = []
    for coords, batch in batch_generator:
        results = model.predict(batch, verbose=0)
        for res, (y, x) in zip(results, coords):
            if res > 0.8:
                objects_coords.append((y, x, res))
        if progress_bar:
            progress_bar.update(batch_size)
    progress_bar.complete()
    return objects_coords


def save_results(source_data: SourceDataV2, results, output_folder, use_img_mask=None):
    if use_img_mask is None:
        use_img_mask = [True for _ in range(len(source_data.images))]
    max_image = source_data.max_image
    dpi = 80
    height, width = max_image.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(max_image, cmap='gray')
    gif_size = 5
    processed = []
    for coord_num, (y, x, probability) in enumerate(results):
        probability = probability[0]
        color = "yellow" if probability <= 0.9 else "green"
        plt.gca().add_patch(Rectangle((x, y), 64, 64,
                                      edgecolor=color,
                                      facecolor='none',
                                      lw=2))
        plt.text(x, y - 10, "{:.2f}".format(probability), color="red", fontsize=15)
        for y_pr, x_pr in processed:
            if x_pr - (gif_size // 2) * 64 <= x <= x_pr + (gif_size // 2) * 64 and \
                    y_pr - (gif_size // 2) * 64 <= y <= y_pr + (gif_size // 2) * 64:
                break
        else:
            processed.append((y, x))
            y_new, x_new, size = get_big_rectangle_coords(y, x, max_image.shape, gif_size)
            plt.gca().add_patch(Rectangle((x_new, y_new), size, size,
                                          edgecolor='red',
                                          facecolor='none',
                                          lw=4))
            plt.text(x_new + 45, y_new + 60, str(len(processed)), color="red", fontsize=40)
            frames = source_data.crop_image(
                source_data.images,
                (y_new, y_new + size),
                (x_new, x_new + size),
                usage_mask=use_img_mask)
            frames = frames * 255
            new_shape = list(frames.shape)
            new_shape[1] += 20
            new_frames = np.zeros(new_shape)
            new_frames[:, :-20, :] = frames
            used_timestamps = []
            for is_used, header in zip(use_img_mask, source_data.headers):
                if is_used:
                    used_timestamps.append(header.timestamp)

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

    not_annotated = plt

    plt.savefig(os.path.join(output_folder, f"results.png"))
    return not_annotated


def annotate_results(source_data: SourceDataV2, plot: plt, output_folder: str):
    start_session_frame_nums = [0]

    start_ts = source_data.headers[0].timestamp
    for num, header in enumerate(source_data.headers[1:], start=1):
        if header.timestamp - start_ts > datetime.timedelta(hours=12):
            start_session_frame_nums.append(num)
            start_ts = header.timestamp
    for start_frame_num in start_session_frame_nums:
        for obj_type in source_data.fetch_known_asteroids_for_image(start_frame_num):
            for item in obj_type:
                print(f"Known object: {item.name}")
                target_x, target_y = item.pixel_coordinates
                target_x = round(float(target_x))
                target_y = round(float(target_y))
                x = (target_x, target_x)
                if target_y < 50:
                    y = (target_y + 4, target_y + 14)
                else:
                    y = (target_y - 4, target_y - 14)
                plot.plot(x, y, color="orange", linewidth=2)

                if target_y < 50:
                    text_y = target_y + 20 + 20
                else:
                    text_y = target_y - 20

                if target_x > source_data.shape[1] - 300:
                    text_x = target_x - 300
                else:
                    text_x = target_x
                plot.text(text_x, text_y, f"{item.name}: {item.magnitude}", color="orange", fontsize=20)
    #
    # plt.axis('off')
    plot.savefig(os.path.join(output_folder, f"results_annotated.png"))
    # logger.log.info(f"Calculations are finished. Check results folder: {output_folder}")
    # if ui_frame:
    #     wx.CallAfter(ui_frame.on_process_finished)


# Decrypt the model weights
def decrypt_model(encrypted_model_path, key=b'J17tdv3zz2nemLNwd17DV33-sQbo52vFzl2EOYgtScw='):
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


def get_model_path():
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


def get_big_rectangle_coords(y, x, image_shape, gif_size):
    size = CHUNK_SIZE
    box_x = 0 if x - size * (gif_size // 2) < 0 else x - size * (gif_size // 2)
    box_y = 0 if y - size * (gif_size // 2) < 0 else y - size * (gif_size // 2)
    image_size_y, image_size_x = image_shape[:2]
    box_x = image_size_x - size * gif_size if x + size * (gif_size // 2 + 1) > image_size_x else box_x
    box_y = image_size_y - size * gif_size if y + size * (gif_size // 2 + 1) > image_size_y else box_y
    return box_y, box_x, size * gif_size
