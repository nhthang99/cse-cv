import numpy as np

from transformation.clip import clip_image


def linear_transformation(image, alpha, beta):
    """Linear transformation

    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image
    alpha : float
        alpha parameter
    beta : float
        beta parameter

    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        transformed image
    """
    return clip_image((alpha * image + beta).astype(np.uint8))


def logarithm_transformation(image, alpha, beta):
    """Logarithm transformation

    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image
    alpha : float
        alpha parameter
    beta : float
        beta parameter

    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        transformed image
    """
    image = ((alpha * np.log2(1 + image/255)) + beta) * 255
    return clip_image(image.astype(np.uint8))


def exponential_transformation(image, alpha, beta):
    """Exponential transformation

    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image
    alpha : float
        alpha parameter
    beta : float
        beta parameter

    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        transformed image
    """
    image = ((alpha * np.exp(image/255)) + beta) * 255
    return clip_image(image.astype(np.uint8))


def power_law_transformation(image, alpha, beta, gamma):
    """Power-law transformation

    Parameters
    ----------
    image : np.ndarray BGR h,w,c-dimentional
        image
    alpha : float
        alpha parameter
    beta : float
        beta parameter
    gamma: float
        gamma parameter

    Returns
    -------
    np.ndarray BGR h,w,c-dimentional
        transformed image
    """
    image = ((alpha * np.power(image/255, gamma)) + beta) * 255
    return clip_image(image.astype(np.uint8))