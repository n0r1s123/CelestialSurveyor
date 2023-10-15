import time

import os
import tqdm
from dataset_creator.dataset_creator import Dataset, SourceData
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from progress.bar import Bar
from xisf import XISF
from multiprocessing import Queue, Process



os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    batch_size = 10
    source_data = SourceData(
        # 'C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\',
        'C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
        'C:\\git\\object_recognition\\star_samples')
    dataset = Dataset(source_data)
    print(dataset.raw_dataset.shape)
    # imgs = dataset.raw_dataset
    imgs = dataset.raw_dataset
    for _ in range(5):
        # imgs = dataset.draw_object_on_image_series_numpy(dataset.raw_dataset)
        imgs = dataset.draw_object_on_image_series_numpy(imgs)
    # for num, item in enumerate(imgs):
    #     item.shape = *item.shape, 1
    #     XISF.write(
    #         os.path.join('C:\\git\\object_recognition\\examples', f"{num:03}.xisf"), item,
    #         creator_app="My script v1.0",
    #         codec='lz4hc', shuffle=True
    #     )

    max_image = dataset.get_max_image(imgs)
    # print(max_image.shape)
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
    # Hide spines, ticks, etc.
    # ax.axis('off')


    imgplot = ax.imshow(max_image, cmap='gray')
    imgs.shape = *imgs.shape, 1

    model = tf.keras.models.load_model(
        'model7.h5'
    )
    total_len = len(ys) * len(xs)
    progress_bar = tqdm.tqdm(total=total_len)
    for y in ys:
        for x in xs:
            shrinked = dataset.get_shrinked_img_series(54, y, x, imgs)
            shrinked = dataset.prepare_images(shrinked)
            shrinked.shape = 1, *shrinked.shape[:-1]
            result = model.predict(shrinked, verbose=0)
            if result > 0.8:
                plt.gca().add_patch(Rectangle((x, y), 54, 54,
                                              edgecolor='green',
                                              facecolor='none',
                                              lw=2))
            progress_bar.update()
    plt.show()

# ======================================================================
# if __name__ == '__main__':
#     batch_size = 10
#     source_data = SourceData('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
#                              'C:\\git\\object_recognition\\star_samples')
#     dataset = Dataset(source_data)
#     print(dataset.raw_dataset.shape)
#
#     imgs = dataset.draw_object_on_image_series_numpy(dataset.raw_dataset)
#     imgs = dataset.draw_object_on_image_series_numpy(imgs)
#     # for num, item in enumerate(imgs):
#     #     item.shape = *item.shape, 1
#     #     XISF.write(
#     #         os.path.join('C:\\git\\object_recognition\\examples', f"{num:03}.xisf"), item,
#     #         creator_app="My script v1.0",
#     #         codec='lz4hc', shuffle=True
#     #     )
#
#     max_image = dataset.get_max_image(imgs)
#     print(max_image.shape)
#     ys = np.arange(0, imgs[0].shape[0], 54)
#     xs = np.arange(0, imgs[0].shape[1], 54)
#
#     dpi = 80
#     height, width = imgs[0].shape
#
#     # What size does the figure need to be in inches to fit the image?
#     figsize = width / float(dpi), height / float(dpi)
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_axes([0, 0, 1, 1])
#     # Hide spines, ticks, etc.
#     # ax.axis('off')
#
#
#     imgplot = ax.imshow(max_image, cmap='gray')
#     imgs.shape = *imgs.shape, 1
#
#     model = tf.keras.models.load_model(
#         'model6.h5'
#     )
#     total_len = len(ys) * len(xs)
#     progress_bar = tqdm.tqdm(total=total_len)
#     shrinked_array = None
#     coords_array = []
#     for y in ys:
#         for x in xs:
#             shrinked = dataset.get_shrinked_img_series(54, y, x, imgs)
#             shrinked = np.array([shrinked[num] - shrinked[0] for num in range(1, len(shrinked))])
#             shrinked[shrinked < 0] = 0
#             shrinked = np.array([(data - np.min(data)) / (np.max(data) - np.min(data)) for data in shrinked])
#             shrinked = shrinked ** 2
#
#             shrinked.shape = 1, *shrinked.shape
#             if shrinked_array is None:
#                 shrinked_array = shrinked
#             else:
#                 shrinked_array = np.append(shrinked_array, shrinked, axis=0)
#             coords_array.append((y, x))
#             # result = model.predict(shrinked, verbose=0)
#             # if result > 0.8:
#             #     plt.gca().add_patch(Rectangle((x, y), 54, 54,
#             #                                   edgecolor='green',
#             #                                   facecolor='none',
#             #                                   lw=2))
#             progress_bar.update()
#
    # batch_size = 10
    # num_of_iterations = len(shrinked_array) // batch_size + (1 if len(shrinked_array % batch_size) > 0 else 0)
    # draw_coords = []
    # progress_bar = tqdm.tqdm(total=num_of_iterations)
    # for num in range(num_of_iterations):
    #     batch_slice = slice(num*batch_size, num*batch_size + batch_size)
    #     batch = shrinked_array[batch_slice]
    #     coords_batch = batch_slice
    #     results_batch = model.predict(batch, verbose=0)
    #     for result, coords in zip(results_batch):
    #         if result > 0.8:
    #             draw_coords.append(coords)
    #     progress_bar.update()
    #
    # for y, x in draw_coords:
    #     plt.gca().add_patch(Rectangle((x, y), 54, 54,
    #                                   edgecolor='green',
    #                                   facecolor='none',
    #                                   lw=2))
    #
    # plt.show()


def gen_batch_coords(coord_pairs, batch_size):
    batch_coords = coord_pairs[:batch_size]
    num = 1
    while batch_coords:
        yield batch_coords
        batch_coords = coord_pairs[batch_size*num: batch_size*(num + 1)]
        num += 1



def worker(queue, imgs, coord_pairs, batch_size):
    iteration_num = len(imgs) // batch_size + 1 if len(imgs) % batch_size else 0
    for num in range(iteration_num):
        batch = np.empty((len(imgs) - 1, 54, 54))
        batch_coord_pairs = coord_pairs[num*batch_size: (num + 1) * batch_size]
        for num1, (y, x) in enumerate(batch_coord_pairs):
            shrinked = dataset.get_shrinked_img_series(54, y, x, imgs)
            shrinked = np.array([shrinked[num] - shrinked[0] for num in range(1, len(shrinked))])
            shrinked[shrinked < 0] = 0
            # shrinked = np.array([(data - np.min(data)) / (np.max(data) - np.min(data)) for data in shrinked])
            shrinked = shrinked ** 2
            # shrinked.shape = 1, *shrinked.shape
            batch[num1] = shrinked
        while queue.full():
            time.sleep(0.1)
        queue.put(batch, batch_coord_pairs)



# if __name__ == '__main__':
#     batch_size = 10
#     workers_num = 1
#     model = tf.keras.models.load_model(
#         'model7.h5'
#     )
#     source_data = SourceData('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
#                              'C:\\git\\object_recognition\\star_samples')
#     dataset = Dataset(source_data)
#     print(dataset.raw_dataset.shape)
#
#     imgs = dataset.draw_object_on_image_series_numpy(dataset.raw_dataset)
#     imgs = dataset.draw_object_on_image_series_numpy(imgs)
#     max_image = dataset.get_max_image(imgs)
#
#     ys = np.arange(0, imgs[0].shape[0], 54)
#     xs = np.arange(0, imgs[0].shape[1], 54)
#     coord_pairs = []
#     for y in ys:
#         for x in xs:
#             coord_pairs.append((y, x))
#
#     # coord_batch_generator = gen_batch_coords(coord_pairs, batch_size)
#     # for num, coord_batch in coord_batch_generator():
#
#     dpi = 80
#     height, width = imgs[0].shape
#
#     # What size does the figure need to be in inches to fit the image?
#     figsize = width / float(dpi), height / float(dpi)
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_axes([0, 0, 1, 1])
#     # Hide spines, ticks, etc.
#     # ax.axis('off')
#
#     imgplot = ax.imshow(max_image, cmap='gray')
#     imgs.shape = *imgs.shape, 1
#
#     processes = []
#     queue = Queue(maxsize=10)
#     for num in range(workers_num):
#         process = Process(target=worker, args=(queue, imgs, coord_pairs[num::workers_num], batch_size))
#         process.start()
#         processes.append(process)
#
#     # Main process does something while workers are running
#     coords_to_print = []
#     while any(process.is_alive() for process in processes):
#         # Do something in the main process
#         # print("Main process is doing something...")
#
#         # Get results from the queue
#         while not queue.empty():
#             shrinked_batch, coord_batch = queue.get()
#             result = model.predict(shrinked_batch, verbose=0)
#             for res, coord in zip(result, coord_batch):
#                 if res > 0.7:
#                     coords_to_print.append(coord)
#             # print(f"Result received: {result}")
#
#
#     # Wait for all processes to finish
#     for process in processes:
#         process.join()
#
#     for y, x in coords_to_print:
#         plt.gca().add_patch(Rectangle((x, y), 54, 54,
#                                       edgecolor='green',
#                                       facecolor='none',
#                                       lw=2))
#
#     plt.show()




