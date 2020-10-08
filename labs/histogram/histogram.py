import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--image', type=str, default="data/image.jpg")
    opt = parse.parse_args()

    image = cv2.imread(opt.image)
    histogram, _ = np.histogram(image, bins=256)
    
    # Draw histogram
    plt.bar(np.arange(256), histogram/(image.shape[0]*image.shape[1]*image.shape[2]))
    plt.show()