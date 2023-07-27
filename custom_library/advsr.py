
import math
import numpy as np
import re
import os

def psnr_batch(img1, img2):
    """
        psnr over a batch
    """
    psnr_final = np.zeros(img1.shape[0])
    if (batch_size == 1):
        temp = psnr_3d_luhong(img1, img2[0])
        psnr_final = temp
    else:
        for i in range (img1.shape[0]):
            temp = psnr_3d_luhong(img1[i], img2[i])
            psnr_final = np.append(psnr_final, temp)
    return psnr_final.mean()

#######################################################

def psnr_3d(img1, img2):

    psnr_final = np.zeros(img1.shape[-1])
    for i in range (img1.shape[-1]):
        #print(type(img1))
        temp = psnr(img1[:,:,i], img2[:,:,i])
        psnr_final = np.append(psnr_final, temp)
    return psnr_final.mean()


def stitch_image(input_images):
    # window shape
    auto_grid = int(math.sqrt(len(input_images)))
    grid = auto_grid
    x_grid = grid
    y_grid = grid  # because test data is 1/4 of the dataset
    z_grid = 1

    x, y = [64, 64]
    hor_pixels = 64 * grid

    total = np.zeros([64, hor_pixels, hor_pixels])

    windows = []
    windows_y = []
    count = 0

    for j in range(y_grid):
        for i in range(x_grid):
            # print(count)
            win_x = [i * 64, (i * 64) + 64]
            win_y = [j * 64, (j * 64) + 64]
            windows.append(win_x)
            windows_y.append(win_y)
            total[:, win_y[0]: win_y[1], win_x[0]: win_x[1]] = input_images[count]
            count += 1
    return total

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)