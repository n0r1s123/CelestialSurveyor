import time

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tqdm
from dataset_creator.dataset_creator import Dataset, SourceData
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image





def get_big_rectangle_coords(y, x, image_shape, gif_size):
    size = 54
    # gif_size = 7  # times of size. Need to use odd numbers

    box_x = 0 if x - size * (gif_size//2) < 0 else x - size * (gif_size//2)
    box_y = 0 if y - size * (gif_size//2) < 0 else y - size * (gif_size//2)
    image_size_y, image_size_x = image_shape[:2]
    box_x = image_size_x - size * gif_size if x + size * (gif_size // 2 + 1) > image_size_x else box_x
    box_y = image_size_y - size * gif_size if y + size * (gif_size // 2 + 1) > image_size_y else box_y
    return box_y, box_x, size * gif_size


if __name__ == '__main__':
    batch_size = 20
    source_data = SourceData(
        [
            'C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
            'C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped\\',
        ],
        'C:\\git\\object_recognition\\star_samples')
    dataset = Dataset(source_data)
    for num, imgs in enumerate(dataset.source_data.raw_dataset):
        print(f"Processing object number {num} of {len(dataset.source_data.raw_dataset)}")
    # imgs = dataset.source_data.raw_dataset[0]
        # for _ in range(5):
        #     imgs, _ = dataset.draw_object_on_image_series_numpy(imgs)

        max_image = dataset.get_max_image(imgs)
        ys = np.arange(0, imgs[0].shape[0], 54)
        ys = ys[:-1]
        xs = np.arange(0, imgs[0].shape[1], 54)
        xs = xs[:-1]

        dpi = 80
        height, width = imgs[0].shape

        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        imgplot = ax.imshow(max_image, cmap='gray')

        model = tf.keras.models.load_model(
            'model20.h5'
        )
        coords = np.array([np.array([y, x]) for y in ys for x in xs])
        number_of_batches = len(coords) // batch_size + (1 if len(coords) % batch_size else 0)
        coord_batches = np.array_split(coords, number_of_batches, axis=0)
        total_len = len(coord_batches)
        progress_bar = tqdm.tqdm(total=total_len)
        objects_coords = []
        for coord_batch in coord_batches:
            imgs_batch = []
            for y, x in coord_batch:
                shrinked = dataset.get_shrinked_img_series(54, y, x, imgs)
                shrinked = dataset.prepare_images(shrinked)
                imgs_batch.append(shrinked)
            imgs_batch = np.array(imgs_batch)
            results = model.predict(imgs_batch, verbose=0)
            for res, (y, x) in zip(results, coord_batch):
                if res > 0.8:
                    objects_coords.append((y, x))
            progress_bar.update()

        gif_size = 7
        processed = []
        for coord_num, (y, x) in enumerate(objects_coords):
            plt.gca().add_patch(Rectangle((x, y), 54, 54,
                                          edgecolor='green',
                                          facecolor='none',
                                          lw=4))
            draw = False
            for y_pr, x_pr in processed:
                if x_pr - (gif_size // 2) * 54 <= x <= x_pr + (gif_size // 2) * 54 and \
                        y_pr - (gif_size // 2) * 54 <= y <= y_pr + (gif_size // 2) * 54:
                    break
            else:
                processed.append((y, x))
                y_new, x_new, size = get_big_rectangle_coords(y, x, imgs[0].shape, gif_size)
                plt.gca().add_patch(Rectangle((x_new, y_new), size, size,
                                              edgecolor='red',
                                              facecolor='none',
                                              lw=6))
                plt.text(x_new + 45, y_new + 60, str(len(processed)), color="red", fontsize=40)

                frames = dataset.get_shrinked_img_series(54 * gif_size, y_new, x_new, imgs)
                frames = frames * 256
                frames = [Image.fromarray(frame).convert('L').convert('P') for frame in frames]
                frames[0].save(f"{num}_{len(processed)}.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)

        plt.show()
