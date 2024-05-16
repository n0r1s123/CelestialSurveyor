from backend.source_data_v2 import SourceDataV2
from progress_bar import ProgressBarCli
from reproject import reproject_interp
import numpy as np
import cv2



if __name__ == '__main__':
    file_list = SourceDataV2.make_file_list("D:\\git\\dataset\\Virgo")

    source_data = SourceDataV2(file_list, to_debayer=True)
    source_data.load_images(progress_bar=ProgressBarCli())
    print(source_data.origional_shape)
    res = source_data.plate_solve_all(progress_bar=ProgressBarCli())
    source_data.align_images_wcs(progress_bar=ProgressBarCli())
    source_data.crop_images()
    source_data.stretch_images(progress_bar=ProgressBarCli())

    for item in source_data.images:
        img = item * (256 * 256 - 1)
        img = img.astype(np.uint16)
        img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
        cv2.imshow("img", img)
        cv2.waitKey(0)


    # for item in source_data.headers:
    #     print(item.wcs)
    # wcs1, star_coords = source_data.plate_solve(0)
    # for idx in range(1, len(source_data.images)):
    #     wcs2, _ = source_data.plate_solve(idx, star_coords)
    #     reprojected, footprint = reproject_interp((np.reshape(source_data.images[idx], source_data.shape[1:3]), wcs2), wcs1, shape_out=source_data.shape[1:3], parallel=True)
    #     img = reprojected * (256 * 256 - 1)
    #     img = img.astype(np.uint16)
    #     img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    #     print(footprint)

    # frames_num = 7
    # headers = [None] * frames_num
    # args = list(zip(range(1, frames_num + 1), headers))
    # print(args)
    # num_per_proc = (frames_num + (3 - frames_num % 3 if frames_num % 3 != 0 else 0)) // 3
    # start = 0
    # while start < frames_num:
    #     end = start + num_per_proc
    #     print(args[start: end])
    #
    #     start = end

