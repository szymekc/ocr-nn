import os
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import random
import numpy as np
from preprocess import binarize, resize, erode, make_frames
from test_display import show_image
import os

split1 = 0.7
split2 = 0.85

if os.path.isfile("./shuffledLabels.pkl"):
    with open("./shuffledLabels.pkl", "rb") as f:
        labels_list = pickle.load(f)
else:
    with open('./dataset/lines.txt', "rt") as f:
        text = list(map(str.split, f.readlines()))
        labels_list = []
        for line in text:
            if " err " not in line:
                labels_list.append([line[0], line[2], line[-1]])
    random.shuffle(labels_list)
    with open("./shuffledLabels.pkl", "wb") as f:
        pickle.dump(labels_list, f)

train_size = int(len(labels_list) * split1)
val_size = int(len(labels_list) * split2)
lab_train = labels_list[:train_size]
lab_val = labels_list[train_size:val_size]
lab_test = labels_list[val_size:]

frame_h = 64
frame_w = 48
frames_num = 128
characters = sorted(set(char for label in np.array(labels_list)[:, 2] for char in label), key=lambda s: sum(map(ord, s)))

char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def preprocess(fname, graylevel, label):
    path_part = fname.split('-')
    path = './dataset/lines/' + path_part[0] + '/' + path_part[0] + '-' + path_part[1] + '/' + fname + '.png'
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    image = binarize(image, graylevel)
    image = resize(image, frame_h)
    image = make_frames(image, frame_h, frame_w, frames_num)
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return image, label


def train_gen():
    random.shuffle(lab_train)
    for fname, graylevel, label in lab_train:
        image, label = preprocess(fname, graylevel, label)
        yield {"images": image, "labels": label}, label


def val_gen():
    random.shuffle(lab_val)
    for fname, graylevel, label in lab_val:
        image, label = preprocess(fname, graylevel, label)
        yield {"images": image, "labels": label}, label


def test_gen():
    random.shuffle(lab_test)
    for fname, graylevel, label in lab_test:
        image, label = preprocess(fname, graylevel, label)
        yield {"images": image, "labels": label}, label
