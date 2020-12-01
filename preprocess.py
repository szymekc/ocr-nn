import cv2
import numpy as np
import tensorflow as tf


def make_frames(image, frame_h, frame_w, frames_num):
    padded = np.full((frame_h, (image.shape[1] // frame_w + 1) * frame_w + frame_w), 255)
    padded[:, :image.shape[1]] = image
    frames = np.full((frames_num, frame_h, frame_w, 1), 255)
    stride = image.shape[1] / frames_num
    for slide in range(frames_num):
        frame = padded[:, int(slide*stride):int(slide*stride) + frame_w]
        frame = np.expand_dims(frame, 2)
        # frames[slide] = tf.image.flip_left_right(tf.image.rot90(frame, k=3))
        frames[slide] = frame
    img = tf.image.convert_image_dtype(frames.astype('uint8'), tf.float32)
    return img


def resize(image, frame_h):
    height = frame_h
    scale = height / image.shape[0]
    width = int(image.shape[1] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def binarize(image, graylevel=128):
    return cv2.threshold(image, int(graylevel), 255, cv2.THRESH_BINARY)[1]

def binarize_adaptive(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def erode(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def denoise(image):
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 15)
