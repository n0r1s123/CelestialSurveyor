import os
import numpy as np
import wx
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Rectangle
import cv2

from cryptography.fernet import Fernet
from io import BytesIO
import h5py
import tensorflow as tf
from PIL import Image

from .source_data import SourceData
from .progress_bar import AbstractProgressBar

from logger.logger import get_logger

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = get_logger()


def find_asteroids(source_data:SourceData, use_img_mask, output_folder, y_splits, x_splits, secondary_alignment: bool, progress_bar: Optional[AbstractProgressBar] = None, model_path="default"):
    if use_img_mask is None:
        use_img_mask = [True for _ in range(len(source_data.images))]

    model_path = get_model_path() if model_path == "default" else model_path
    logger.log.debug(f"Loading model: {model_path}")
    model = decrypt_model(model_path)
    batch_size = 10

    if not secondary_alignment:
        y_splits = 1
        x_splits = 1
    source_data.chop_imgs(y_splits=y_splits, x_splits=x_splits)
    splits = source_data.gen_splits(y_splits, x_splits, secondary_alignment, use_img_mask)
    objects_coords = []
    for imgs, y_offset, x_offset, use_img_mask in splits:
        ys = np.arange(0, imgs[0].shape[0], 64)
        ys[-1] = imgs[0].shape[0] - 64 - source_data.BOARDER_OFFSET - 1

        xs = np.arange(0, imgs[0].shape[1], 64)
        xs[-1] = imgs[0].shape[1] - 64 - source_data.BOARDER_OFFSET - 1
        coords = np.array([np.array([y, x]) for y in ys for x in xs])

        number_of_batches = len(coords) // batch_size + (1 if len(coords) % batch_size else 0)
        progress_bar_len = number_of_batches * y_splits * x_splits

        if progress_bar:
            progress_bar.set_total(progress_bar_len)
        coord_batches = np.array_split(coords, number_of_batches, axis=0)

        for coord_batch in coord_batches:
            imgs_batch = []
            ts_pred = []
            for y, x in coord_batch:
                shrinked = source_data.get_shrinked_img_series(imgs, 64, y, x, use_img_mask)
                shrinked = source_data.prepare_images(shrinked)
                timestamps = source_data.timestamps
                new_timestamps = []
                for ts, checked in zip(timestamps, use_img_mask):
                    if checked:
                        new_timestamps.append(ts)
                timestamps = new_timestamps
                shrinked, timestamps = source_data.adjust_series_to_min_len(shrinked, timestamps)
                timestamps = source_data.prepare_timestamps(timestamps)
                timestamps = np.swapaxes(timestamps, 0, 1)
                imgs_batch.append(shrinked)
                ts_pred.append(timestamps)
            imgs_batch = np.array(imgs_batch)
            results = model.predict([imgs_batch, np.array(ts_pred)], verbose=0)
            for res, (y, x) in zip(results, coord_batch):
                if res > 0.9:
                    objects_coords.append((y + y_offset, x + x_offset, res))
            if progress_bar:
                progress_bar.update()

    ########
    # Save results
    ########
    max_image = source_data.get_max_image(use_img_mask)
    dpi = 80
    height, width = max_image.shape
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(max_image, cmap='gray')
    gif_size = 5
    processed = []
    print(objects_coords)
    for coord_num, (y, x, probability) in enumerate(objects_coords):
        # confirmed = confirm_prediction(model, dataset=dataset, y=y, x=x, dataset_num=num)
        # if hide_unconfirmed and not confirmed:
        #     continue
        # color = 'green' if confirmed else 'yellow'
        probability = probability[0]
        color = "yellow" if probability <= 0.95 else "green"
        plt.gca().add_patch(Rectangle((x, y), 64, 64,
                                      edgecolor=color,
                                      # edgecolor='green',
                                      facecolor='none',
                                      lw=4))
        plt.text(x, y - 10, "{:.2f}".format(probability), color="red", fontsize=20)
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
                                          lw=6))
            plt.text(x_new + 45, y_new + 60, str(len(processed)), color="red", fontsize=40)

            frames = source_data.get_shrinked_img_series(source_data.images, 64 * gif_size, y_new, x_new, use_img_mask)
            frames = frames * 256
            new_shape = list(frames.shape)
            new_shape[1] += 20
            new_frames = np.zeros(new_shape)
            new_frames[:, :-20, :] = frames
            used_timestamps = []
            for is_used, (_, ts, _) in zip(use_img_mask, source_data.timestamped_file_list):
                if is_used:
                    used_timestamps.append(ts)

            for frame, original_ts in zip(new_frames, used_timestamps):
                cv2.putText(frame, text=original_ts.strftime("%d/%m/%Y %H:%M:%S %Z"), org=(70, 64 * gif_size + 16),
                            fontFace=1, fontScale=1, color=(255, 255, 255), thickness=0)
            new_frames = [Image.fromarray(frame).convert('L').convert('P') for frame in new_frames]
            new_frames[0].save(
                os.path.join(output_folder, f"{len(processed)}.gif"),
                save_all=True,
                append_images=new_frames[1:],
                duration=200,
                loop=0)

    plt.savefig(os.path.join(output_folder, f"results.png"))


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
        secondary_dir = os.path.join(root_dir, '_internal')
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
    size = 64
    box_x = 0 if x - size * (gif_size // 2) < 0 else x - size * (gif_size // 2)
    box_y = 0 if y - size * (gif_size // 2) < 0 else y - size * (gif_size // 2)
    image_size_y, image_size_x = image_shape[:2]
    box_x = image_size_x - size * gif_size if x + size * (gif_size // 2 + 1) > image_size_x else box_x
    box_y = image_size_y - size * gif_size if y + size * (gif_size // 2 + 1) > image_size_y else box_y
    return box_y, box_x, size * gif_size

