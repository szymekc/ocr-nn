import cv2
import numpy as np
import pickle
from dataset import load_dataset
import tensorflow as tf
from tensorflow.keras import layers
from dataset import split_dataset
import itertools
import random

images = pickle.load(open("images.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))
preprocessed_x_train = pickle.load(open("preprocessed_x_train.pkl", "rb"))
preprocessed_y_train = pickle.load(open("preprocessed_y_train.pkl", "rb"))
preprocessed_x_val = pickle.load(open("preprocessed_x_val.pkl", "rb"))
preprocessed_y_val = pickle.load(open("preprocessed_y_val.pkl", "rb"))
preprocessed_x_test = pickle.load(open("preprocessed_x_test.pkl", "rb"))
preprocessed_y_test = pickle.load(open("preprocessed_y_test.pkl", "rb"))

frame_size = 128
characters = sorted(set(char for label in labels for char in label), key=lambda s: sum(map(ord, s)))

char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(images, labels)
# preprocessed_x_train = preprocessed_x_train[:100]
# preprocessed_y_train = preprocessed_y_train[:100]
# preprocessed_x_val = preprocessed_x_val[:10]
# preprocessed_y_val = y_val[:10]


def make_frames(image):
    padded = np.full((frame_size, (image.shape[1] // frame_size + 1) * frame_size + frame_size), 255)
    padded[:, :image.shape[1]] = image
    frames = np.full((100, frame_size, frame_size, 1), 255)
    stride = image.shape[1] / 100
    for slide in range(100):
        frame = padded[:, int(slide*stride):int(slide*stride) + frame_size]
        frame = np.expand_dims(frame, 2)
        frames[slide] = tf.image.flip_left_right(tf.image.rot90(frame, k=3))
    img = tf.image.convert_image_dtype(frames.astype('uint8'), tf.float32)
    return img


def resize(image):
    height = frame_size
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

def sample_train():
    rand = list(zip(preprocessed_x_train, preprocessed_y_train))
    random.shuffle(rand)
    randx, randy = zip(*rand)
    for i in range(len(randx)):
        image = randx[i]
        image = make_frames(image)
        label = randy[i]
        yield {"images": image, "labels": label}, label


def sample_val():
    rand = list(zip(preprocessed_x_val, preprocessed_y_val))
    random.shuffle(rand)
    randx, randy = zip(*rand)
    for i in range(len(randx)):
        image = randx[i]
        image = make_frames(image)
        label = randy[i]
        yield {"images": image, "labels": label}, label


def sample_test():
    rand = list(zip(preprocessed_x_test, preprocessed_y_test))
    random.shuffle(rand)
    randx, randy = zip(*rand)
    for i in range(len(randx)):
        image = randx[i]
        image = make_frames(image)
        label = randy[i]
        yield {"images": image, "labels": label}, label

def preprocess_all_data(images, labels):
    new_images = []
    new_labels = []
    for image in images:
        image = denoise(image)
        image = binarize(image)
        image = resize(image)
        image = thin(image)
        new_images.append(image)
    for label in labels:
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        new_labels.append(label)
    return new_images, new_labels
