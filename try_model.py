import time

import os
import tqdm
from dataset_creator.dataset_creator import Dataset, SourceData
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    batch_size = 20
    source_data = SourceData(
        [
            # 'C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
            # 'C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\',
            # 'C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped\\',
        ],
        'C:\\git\\object_recognition\\star_samples')
    dataset = Dataset(source_data)
    imgs = dataset.source_data.raw_dataset[0]
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
        'model15.h5'
    )
    coords = np.array([np.array([y, x]) for y in ys for x in xs])
    number_of_batches = len(coords) // batch_size + (1 if len(coords) % batch_size else 0)
    coord_batches = np.array_split(coords, number_of_batches, axis=0)
    total_len = len(coord_batches)
    progress_bar = tqdm.tqdm(total=total_len)
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
                plt.gca().add_patch(Rectangle((x, y), 54, 54,
                                              edgecolor='green',
                                              facecolor='none',
                                              lw=4))
        progress_bar.update()
    plt.show()
