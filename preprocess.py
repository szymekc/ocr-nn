import cv2
import numpy as np
import pickle
from dataset import load_dataset
import tensorflow as tf
from tensorflow.keras import layers
from dataset import split_dataset
import itertools


images = pickle.load(open("images.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))

characters = sorted(set(char for label in labels for char in label), key=lambda s: sum(map(ord, s)))

char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

x_train, y_train, x_test, y_test = split_dataset(images, labels)


def make_frames(image):
    padded = np.full((128, (image.shape[1] // 128 + 1) * 128 + 128), 255)
    padded[:, :image.shape[1]] = image
    frames = np.full((100, 128, 128, 1), 255)
    stride = image.shape[1] / 100
    for slide in range(100):
        frame = padded[:, int(slide*stride):int(slide*stride) + 128]
        frame = np.expand_dims(frame, 2)
        frames[slide] = tf.image.flip_left_right(tf.image.rot90(frame, k=3))
    return tf.image.convert_image_dtype(frames.astype('uint8'), tf.float32)


def resize(image):
    height = 128
    scale = height / image.shape[0]
    width = int(image.shape[1] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def binarize(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def thin(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def denoise(image):
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 15)


def preprocess_sample_train():
    for i in itertools.count(0):
        image = x_train[i]
        label = y_train[i]
        image = denoise(image)
        image = binarize(image)
        image = resize(image)
        image = thin(image)
        image = make_frames(image)
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        yield {"images": image, "labels": label}, label


def preprocess_sample_test():
    for i in itertools.count(0):
        image = x_test[i]
        label = y_test[i]
        image = denoise(image)
        image = binarize(image)
        image = resize(image)
        image = thin(image)
        image = make_frames(image)
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        yield {"images": image, "labels": label}, label

