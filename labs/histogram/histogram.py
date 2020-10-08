import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(image):
    """Plot histogram of image

    Parameters
    ----------
    image : np.ndarray BGR h,w,c dimentional
        image
    """
    hist, _ = np.histogram(image, bins=256)
    plt.bar(np.arange(256), hist)
    plt.show()


def histogram_equalization(image):
    """Histogram Equalization

    Parameters
    ----------
    image : np.ndarray BGR h,w,c dimentional
        image

    Returns
    -------
    np.ndarray BGR h,w,c dimentional
        equalized image
    """
    histogram, _ = np.histogram(image, bins=256)

    cdf = (histogram / histogram.sum()).cumsum()
    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())
    cdf_m *= 255
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    eq_image = cdf[image]

    return eq_image


def convert_histogram(src_image, dest_image):
    """Convert image to histogram of other image

    Parameters
    ----------
    src_image : np.ndarray BGR h,w,c dimentional
        source image
    dest_image : np.ndarray BGR h,w,c dimentional
        destination image

    Returns
    -------
    np.ndarray BGR h,w,c dimentional
        equalized image
    """
    src_hist, _ = np.histogram(src_image, bins=256)
    dst_hist, _ = np.histogram(dest_image, bins=256)

    src_cdf = (src_hist / src_hist.sum()).cumsum()
    dst_cdf = (dst_hist / dst_hist.sum()).cumsum()

    cdf = np.zeros_like(src_cdf)

    for i in range(256):
        idx = np.argmin(np.abs(src_cdf - dst_cdf[i]))
        cdf[idx] = i

    return cdf[src_image]


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--image', type=str, default="data/image.jpg")
    parse.add_argument('-d', '--dst-image', type=str, default="data/luna.jpg")
    opt = parse.parse_args()

    image = cv2.imread(opt.image)
    plot_histogram(image)
    
    eq_image = histogram_equalization(image)
    plot_histogram(eq_image)

    dst_image = cv2.imread(opt.dst_image)
    plot_histogram(dst_image)
    
    eq_image = convert_histogram(image, dst_image)
    plot_histogram(eq_image)
