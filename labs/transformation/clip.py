import numpy as np


def clip_image(image):
    """Clip image

    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image

    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        clipped image
    """
    return np.clip(image, 0, 255)