import keras
import tensorflow as tf
from keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from dataset import load_dataset
from keras_ctcmodel import CTCModel as ctc


class OcrModel(object):
    def __init__(self, train, test):
        inputs = layers.Input(shape=(None, 128, 128, 1), name="images", dtype="float32")
        labels = layers.Input(name="labels", shape=(None,), dtype="float32")
        conv_1 = TimeDistributed(layers.Conv2D(16, 5, activation='relu'))(inputs)
        pool_1 = TimeDistributed(layers.MaxPooling2D())(conv_1)
        batch_norm_1 = TimeDistributed(layers.BatchNormalization())(pool_1)

        conv_2 = TimeDistributed(layers.Conv2D(32, 5, activation='relu'))(batch_norm_1)
        pool_2 = TimeDistributed(layers.MaxPooling2D())(conv_2)
        batch_norm_2 = TimeDistributed(layers.BatchNormalization())(pool_2)

        conv_3 = TimeDistributed(layers.Conv2D(64, 3, activation='relu'))(batch_norm_2)
        drop_1 = TimeDistributed(layers.Dropout(0.2))(conv_3)
        batch_norm_3 = TimeDistributed(layers.BatchNormalization())(drop_1)

        conv_4 = TimeDistributed(layers.Conv2D(128, 3, activation='relu'))(batch_norm_3)
        drop_2 = TimeDistributed(layers.Dropout(0.2))(conv_4)
        batch_norm_4 = TimeDistributed(layers.BatchNormalization())(drop_2)
        flatten_1 = TimeDistributed(layers.Flatten())(batch_norm_4)

        blstm_1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(flatten_1)
        blstm_2 = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(blstm_1)

        drop_3 = layers.Dropout(0.5)(blstm_2)
        dense_1 = layers.Dense(256)(drop_3)
        softmax = layers.Softmax()(dense_1)

        opt = keras.optimizers.Adam()
        self.model.compile(opt)
        self.history = self.model.fit(train, batch_size=32, epochs=100)
        self.results = self.model.evaluate(test, batch_size=32)
