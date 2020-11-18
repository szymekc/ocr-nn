import os
import cv2
import pickle
import tensorflow as tf


def load_dataset(path):
    with open(path + '/lines.txt', "rt") as f:
        text = list(map(str.split, f.readlines()))
        labels_dict = {}
        labels = []
        images = []
        for line in text:
            if "err" not in line:
                labels_dict[line[0]] = line[-1]
    for root, dirs, files in os.walk('./dataset/lines'):
        dirs.sort()
        files.sort()
        for f in files:
            try:
                labels.append(labels_dict[f.replace('.png', '')])
                images.append(cv2.cvtColor(cv2.imread(root + '/' + f), cv2.COLOR_BGR2GRAY))
            except KeyError:
                continue
    pickle.dump(images, open("images.pkl", "wb"))
    pickle.dump(labels, open("labels.pkl", "wb"))
    return images, labels


def split_dataset(images, labels, split=0.9):
    train_size = int(len(images) * split)
    x_train, y_train = images[:train_size], labels[:train_size]
    x_test, y_test = images[train_size:], labels[train_size:]
    return x_train, y_train, x_test, y_test


def conv_to_tf(images_train, labels_train, images_test, labels_test):
    train_set = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    test_set = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
    return train_set, test_set

