import cv2
import numpy as np
from tensorflow.keras import layers


def resize(images):
    resized = []
    height = 128
    for image in images:
        scale = height / image.shape[0]
        width = int(image.shape[1] * scale)
        resized.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    return resized


def binarize(images):
    binarized = []
    for image in images:
        binarized.append(cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    return binarized


def thin(images):
    thinned = []
    kernel = np.ones((5, 5), np.uint8)
    for image in images:
        thinned.append(cv2.erode(image, kernel, iterations=1))
    return thinned


def denoise(images):
    denoised = []
    for image in images:
        denoised.append(cv2.fastNlMeansDenoising(image, None, 10, 7, 15))
    return denoised


def preprocess_all(images, labels):
    images = denoise(images)
    images = binarize(images)
    images = resize(images)
    images = thin(images)
    return images
