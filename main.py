from model import OcrModel
import pickle
import tensorflow as tf
import numpy as np
import dataset
import itertools
from keras_preprocessing import sequence
from tensorflow.keras import layers

from dataset import make_frames


x = pickle.load(open('data_preprocessed.pkl', "rb"))
y = pickle.load(open('labels.pkl', "rb"))
characters = sorted(set(char for label in y for char in label), key=lambda s: sum(map(ord, s)))
# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


x_train, y_train, x_test, y_test = dataset.split_dataset(x, y)
num_train = len(x_train)
num_test = len(x_test)


def train_gen():
    for i in itertools.count(0):
        yield make_frames(x_train[i]), char_to_num(tf.strings.unicode_split(y_train[i], input_encoding='UTF-8'))


def test_gen():
    for i in itertools.count(0):
        yield make_frames(x_test[i]), char_to_num(tf.strings.unicode_split(y_test[i], input_encoding='UTF-8'))


x_train_batches = tf.data.Dataset.from_generator(
    train_gen,
    (tf.float32, tf.int32)
).padded_batch(32,
               padded_shapes=([None, None, None, None], [None]),
               padding_values=(None, -1))

model = OcrModel(len(characters))
history = model.model.fit(x_train_batches, batch_size=32, epochs=100)
#results = model.model.evaluate(x=[x_test, y_test, x_test_len, y_test_len], batch_size=32)
print('dupa')
