import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        # self.loss_fn = keras.backend.ctc_batch_cost
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def cnn(inputs):
    # x = layers.Conv2D(16, 5, activation='relu')(inputs)
    # x = layers.MaxPooling2D()(x)
    # x = layers.BatchNormalization()(x)
    #
    # x = layers.Conv2D(32, 5, activation='relu')(x)
    # x = layers.MaxPooling2D()(x)
    # x = layers.BatchNormalization()(x)
    #
    # x = layers.Conv2D(64, 3, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.BatchNormalization()(x)
    #
    # x = layers.Conv2D(128, 3, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.BatchNormalization()(x)
    # out = layers.Dense(256, activation="relu")(x)
    # return out
    x = TimeDistributed(layers.Conv2D(8, 3, activation='relu'), name="conv1")(inputs)
    x = TimeDistributed(layers.MaxPooling2D(), name="pool1")(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm1")(x)

    x = TimeDistributed(layers.Conv2D(16, 3, activation='relu'), name="conv2")(x)
    x = TimeDistributed(layers.MaxPooling2D(), name="pool2")(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm2")(x)

    x = TimeDistributed(layers.Conv2D(32, 3, activation='relu'), name="conv3")(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop1")(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm3")(x)

    x = TimeDistributed(layers.Conv2D(64, 3, activation='relu'), name="conv4")(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop2")(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm4")(x)

    x = TimeDistributed(layers.Conv2D(128, 3, activation='relu'), name="conv5")(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop3")(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm5")(x)
    x = TimeDistributed(layers.Flatten(), name="flatten")(x)
    out = TimeDistributed(layers.Dense(256, activation="relu"))(x)
    return out


def rnn(inputs, charset_len):
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.Dropout(0.5)(x)
    out = layers.Dense(charset_len + 1, activation='softmax')(x)
    return out

class OcrModel(object):
    def __init__(self, charset_len, batch_size, frame_size):
        inputs = layers.Input(shape=[None, frame_size, frame_size, 1], name="images", dtype="float32", batch_size=batch_size)
        labels = layers.Input(shape=[None], dtype="float32", name="labels", batch_size=batch_size)

        cnn_out = cnn(inputs)

        rnn_out = rnn(cnn_out, charset_len)
        output = CTCLayer(name="ctc_loss")(labels, rnn_out)

        opt = tf.keras.optimizers.SGD(lr=0.0001)
        self.model = tf.keras.models.Model(inputs={"images": inputs, "labels": labels}, outputs=[output], name="ocr_nn_v1")
        print(self.model.summary(line_length=150))
        self.model.compile(opt)
