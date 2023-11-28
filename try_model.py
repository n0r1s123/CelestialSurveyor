import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tqdm
from dataset_creator.dataset_creator import Dataset, SourceData
import tensorflow as tf
tf.get_logger().setLevel('FATAL')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import cv2
from cryptography.fernet import Fernet
from io import BytesIO
import h5py
import argparse


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
    box_x = 0 if x - size * (gif_size//2) < 0 else x - size * (gif_size//2)
    box_y = 0 if y - size * (gif_size//2) < 0 else y - size * (gif_size//2)
    image_size_y, image_size_x = image_shape[:2]
    box_x = image_size_x - size * gif_size if x + size * (gif_size // 2 + 1) > image_size_x else box_x
    box_y = image_size_y - size * gif_size if y + size * (gif_size // 2 + 1) > image_size_y else box_y
    return box_y, box_x, size * gif_size


def confirm_prediction(model, dataset, y, x, dataset_num):
    movement = 5  # pixels
    directions = (-2, -1, 0, 1, 2)
    coords = np.array([[y + dir_y * movement, x + dir_x * movement] for dir_y in directions for dir_x in directions])
    imgs_batch = []
    ts_pred = []
    ts_diff = dataset.source_data.diff_timestamps[dataset_num]
    ts_norm = dataset.source_data.normalized_timestamps[dataset_num][1:]
    ts = np.array(list(zip(ts_diff, ts_norm)))
    y_shape, x_shape =  dataset.source_data.raw_dataset[dataset_num][0].shape[:2]
    for y, x in coords:
        if x < 0 or y < 0 or x + 64 > x_shape or y + 64 > y_shape:
            continue
        shrinked = dataset.get_shrinked_img_series(64, y, x, dataset_idx=dataset_num)
        shrinked = dataset.prepare_images(shrinked)
        imgs_batch.append(shrinked)
        ts_pred.append(ts)
    imgs_batch = np.array(imgs_batch)
    results = model.predict([imgs_batch, np.array(ts_pred)], verbose=0)
    number_of_found_objects = (results > 0.80).sum()
    return number_of_found_objects > 2


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


def main(source_folder, output_folder, model_path, hide_unconfirmed, non_linear):
    if not os.path.exists(source_folder):
        raise ValueError(f"Path '{source_folder}' doesn't exist. Please check the path specified")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    batch_size = 20
    source_data = SourceData(
        [
            source_folder,
            # 'C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\',
            # # 'C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
            # 'C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\',
            # 'C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped\\',
            # 'C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1\\',
            # 'C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped\\',
            # 'C:\\Users\\bsolomin\\Astro\\Orion\\Part_two\\cropped\\',
            # 'C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\',
            # # 'C:\\Users\\bsolomin\\Astro\\M81\\cropped\\',
            # # 'D:\\Boris\\astro\\M31_2\\Pix\\registered\\Light_BIN-1_EXPOSURE-120.00s_FILTER-NoFilter_RGB',
            # # 'D:\\Boris\\astro\\M78\\Pix\\registered\\Light_BIN-1_EXPOSURE-120.00s_FILTER-NoFilter_RGB'
        ],
        'C:\\git\\object_recognition\\star_samples',
        non_linear=non_linear)
    dataset = Dataset(source_data)
    for num, imgs in enumerate(dataset.source_data.raw_dataset):
        print(f"Processing object number {num+1} of {len(dataset.source_data.raw_dataset)}")
        # imgs = dataset.source_data.raw_dataset[0]
        # for _ in range(5):
        #     imgs, _ = dataset.draw_object_on_image_series_numpy(imgs)

        max_image = dataset.get_max_image(imgs)
        ys = np.arange(0, imgs[0].shape[0], 64)
        ys = ys[:-1]
        xs = np.arange(0, imgs[0].shape[1], 64)
        xs = xs[:-1]

        dpi = 80
        height, width = imgs[0].shape

        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        imgplot = ax.imshow(max_image, cmap='gray')

        if model_path == "default":
            model_path = get_model_path()


        model = decrypt_model(model_path)


        coords = np.array([np.array([y, x]) for y in ys for x in xs])
        number_of_batches = len(coords) // batch_size + (1 if len(coords) % batch_size else 0)
        coord_batches = np.array_split(coords, number_of_batches, axis=0)
        total_len = len(coord_batches)

        objects_coords = []
        ts_diff = dataset.source_data.diff_timestamps[num]
        ts_norm = dataset.source_data.normalized_timestamps[num][1:]
        ts = np.array(list(zip(ts_diff, ts_norm)))
        progress_bar = tqdm.tqdm(total=total_len)
        for coord_batch in coord_batches:
            imgs_batch = []
            ts_pred = []
            for y, x in coord_batch:
                shrinked = dataset.get_shrinked_img_series(64, y, x, imgs)
                shrinked = dataset.prepare_images(shrinked)
                imgs_batch.append(shrinked)
                ts_pred.append(ts)

            imgs_batch = np.array(imgs_batch)
            # ts_batch = dataset.source_data.diff_timestamps[num]
            results = model.predict([imgs_batch, np.array(ts_pred)], verbose=0)
            for res, (y, x) in zip(results, coord_batch):
                if res > 0.8:
                    objects_coords.append((y, x))
            progress_bar.update()

        gif_size = 5
        processed = []
        for coord_num, (y, x) in enumerate(objects_coords):
            confirmed = confirm_prediction(model, dataset=dataset, y=y, x=x, dataset_num=num)
            if hide_unconfirmed and not confirmed:
                continue
            color = 'green' if confirmed else 'yellow'
            plt.gca().add_patch(Rectangle((x, y), 64, 64,
                                          edgecolor=color,
                                          facecolor='none',
                                          lw=4))
            draw = False
            for y_pr, x_pr in processed:
                if x_pr - (gif_size // 2) * 64 <= x <= x_pr + (gif_size // 2) * 64 and \
                        y_pr - (gif_size // 2) * 64 <= y <= y_pr + (gif_size // 2) * 64:
                    break
            else:
                processed.append((y, x))
                y_new, x_new, size = get_big_rectangle_coords(y, x, imgs[0].shape, gif_size)
                plt.gca().add_patch(Rectangle((x_new, y_new), size, size,
                                              edgecolor='red',
                                              facecolor='none',
                                              lw=6))
                plt.text(x_new + 45, y_new + 60, str(len(processed)), color="red", fontsize=40)

                frames = dataset.get_shrinked_img_series(64 * gif_size, y_new, x_new, imgs)
                frames = frames * 256
                new_shape = list(frames.shape)
                new_shape[1] += 20
                new_frames = np.zeros(new_shape)
                new_frames[:, :-20, :] = frames
                for frame, original_ts in zip(new_frames, dataset.source_data.timestamps[num]):
                    cv2.putText(frame, text=original_ts.strftime("%d/%m/%Y %H:%M:%S %Z"), org=(70, 64 * gif_size + 16),
                                fontFace=1, fontScale=1, color=(255, 255, 255), thickness=0)
                new_frames = [Image.fromarray(frame).convert('L').convert('P') for frame in new_frames]
                new_frames[0].save(
                    os.path.join(output_folder, f"{num}_{len(processed)}.gif"),
                    save_all=True,
                    append_images=new_frames[1:],
                    duration=200,
                    loop=0)

        plt.savefig(os.path.join(output_folder, f"results{num}.png"))


if __name__ == '__main__':
    version = "0.1.2"
    arg_parser = argparse.ArgumentParser(
        prog='CelestialSurveyor',
        description='It\'s is designed to analyze astronomical images with the primary goal of identifying and '
                    'locating asteroids and comets within the vastness of the cosmic terrain')
    arg_parser.add_argument('-s', '--source_folder', dest='source_folder', type=str,
                            help='Path to the folder with xisf files to be analyzed')
    arg_parser.add_argument('-o', '--output_folder', dest='output_folder', type=str,
                            help='Path to the folder where results will be stored')
    arg_parser.add_argument('-m', '--model_path', dest='model_path', type=str, default="default",
                            help='Path to the AI model file')
    arg_parser.add_argument('-u', '--hide_unconfirmed', dest='hide_unconfirmed', action="store_true",
                            help='Path to the folder where results will be stored')
    arg_parser.add_argument('-n', '--non_linear', dest='non_linear', action="store_true",
                            help='Provide this key if the images are not in linear state')
    arg_parser.add_argument('-v', '--version', dest='version', action="store_true",
                            help='Display version of this app.')
    provided_args = arg_parser.parse_args()
    if provided_args.version:
        print(f"CelestialSurveyor v{version}")
    else:
        main(
            source_folder=provided_args.source_folder,
            output_folder=provided_args.output_folder,
            model_path=provided_args.model_path,
            hide_unconfirmed=provided_args.hide_unconfirmed,
            non_linear=provided_args.non_linear,
        )
