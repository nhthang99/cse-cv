import os
import sys
import shutil
import argparse

import cv2

PYTHON_PATH = os.path.abspath('.')
sys.path.insert(0, PYTHON_PATH)

from transformation.transform import *


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--image', type=str, default="data/luna.jpg")
    parse.add_argument('-s', '--show', action='store_true')
    parse.add_argument('-o', '--output', type=str, default="output/")
    opt = parse.parse_args()

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    else:
        shutil.rmtree(opt.output)
        os.makedirs(opt.output)

    # Read image
    image = cv2.imread(opt.image, 0)

    # Transformation
    linear_image = linear_transformation(image.copy(), 0.5, 10)
    logarith_image = logarithm_transformation(image.copy(), 1.5, 10)
    exponential_image = exponential_transformation(image.copy(), 1, -1)
    power_law_image = power_law_transformation(image.copy(), 1, -1, 0.5)

    # Save image
    linear_path = os.path.join(opt.output, "linear" + os.path.splitext(opt.image)[-1])
    cv2.imwrite(linear_path, linear_image)
    logarith_path = os.path.join(opt.output, "logarithm" + os.path.splitext(opt.image)[-1])
    cv2.imwrite(logarith_path, logarith_image)
    exponential_path = os.path.join(opt.output, "exponential" + os.path.splitext(opt.image)[-1])
    cv2.imwrite(exponential_path, exponential_image)
    power_law_path = os.path.join(opt.output, "power_law" + os.path.splitext(opt.image)[-1])
    cv2.imwrite(power_law_path, power_law_image)


    if opt.show:
        # Show Linear Transformation
        img = cv2.hconcat([image, linear_image])
        cv2.imshow("Linear Transformation", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Linear Transformation")

        # Show Logarithm Transformation
        img = cv2.hconcat([image, logarith_image])
        cv2.imshow("Logarithm Transformation", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Logarithm Transformation")

        # Show Exponential Transformation
        img = cv2.hconcat([image, exponential_image])
        cv2.imshow("Exponential Transformation", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Exponential Transformation")

        # Show Power-law Transformation
        img = cv2.hconcat([image, power_law_image])
        cv2.imshow("Power-law Transformation", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Power-law Transformation")
