import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from dataset import characters, num_to_char
from metrics import CERMetric, WERMetric, LERMetric

word_beam_search_module = tf.load_op_library('./ctcwbs/cpp/proj/tf/TFWordBeamSearch.so')

with open('words.txt', 'rt') as f:
    corpus = " ".join(f.readlines())
words = "".join(characters)
word_chars = "'-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

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
        # input_shape = K.shape(y_pred)
        # input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')
        # y_pred = K.ctc_decode(y_pred, input_length, greedy=False, beam_width=10)
        y_pred = tf.transpose(y_pred, [1, 0, 2])
        y_pred = word_beam_search_module.word_beam_search(y_pred, 25, 'Words', 0, corpus, words, word_chars)
        # tf.print("Predicted line: ", num_to_char(y_pred), summarize=-1)
        # tf.print("True line:", num_to_char(y_true), summarize=-1)
        # At test time, just return the computed predictions
        return y_pred


def cnn(inputs):
    x = TimeDistributed(layers.Conv2D(8, 3, padding="valid"), name="conv1")(inputs)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = TimeDistributed(layers.MaxPooling2D(), name="pool1")(x)
    x = layers.BatchNormalization()(x)

    x = TimeDistributed(layers.Conv2D(16, 3, padding="valid"), name="conv2")(x)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = TimeDistributed(layers.MaxPooling2D(), name="pool2")(x)
    x = layers.BatchNormalization()(x)

    x = TimeDistributed(layers.Conv2D(32, 3, padding="valid"), name="conv3")(x)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = layers.BatchNormalization()(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop1")(x)

    x = TimeDistributed(layers.Conv2D(64, 3, padding="valid"), name="conv4")(x)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = layers.BatchNormalization()(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop2")(x)

    x = TimeDistributed(layers.Conv2D(128, 3, padding="valid"), name="conv5")(x)
    x = TimeDistributed(layers.MaxPooling2D(pool_size=(2, 1)))(x)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = layers.BatchNormalization()(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop3")(x)
    x = TimeDistributed(layers.Flatten(), name="flatten")(x)
    out = TimeDistributed(layers.Dense(512, activation="relu"))(x)

    return out


def rnn(inputs, charset_len):
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(charset_len + 1, activation='softmax', name="dense_2")(x)
    return out

class OcrModel(object):
    def __init__(self, charset_len, batch_size, frame_h, frame_w, learning_rate=0.0003):
        inputs = layers.Input(shape=[None, frame_h, frame_w, 1], name="images", dtype="float32", batch_size=batch_size)
        labels = layers.Input(shape=[None], dtype="int32", name="labels", batch_size=batch_size)

        cnn_out = cnn(inputs)

        rnn_out = rnn(cnn_out, charset_len)
        output = CTCLayer(name="ctc_loss")(labels, rnn_out)

        opt = tf.keras.optimizers.Adam(learning_rate)
        self.model = tf.keras.models.Model(inputs={"images": inputs, "labels": labels}, outputs=[output], name="ocr_nn_v1")
        print(self.model.summary(line_length=150))
        self.model.compile(opt, metrics=[CERMetric(), WERMetric(), LERMetric()])
