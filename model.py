import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from preprocess import characters, num_to_char
import tensorflow.keras.backend as K
import re

word_beam_search_module = tf.load_op_library('./ctcwbs/cpp/proj/tf/TFWordBeamSearch.so')

with open('wordlist.txt', 'rt') as f:
    corpus = " ".join(f.readlines())
words = "".join(characters)
word_chars = "'-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def dense_to_sparse(dense_tensor, sparse_val=0):
    """Inverse of tf.sparse_to_dense.

    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.
    Returns:
        SparseTensor equivalent to the dense input.
    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)

class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """

    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        #
        # decode, log = K.ctc_decode(y_pred,
        #                            input_length,
        #                            greedy=False, beam_width=16)
        # decode = tf.transpose(decode, [1, 0])
        # decode = K.ctc_label_dense_to_sparse(decode, K.cast(input_length, 'int32'))
        decode = K.cast(dense_to_sparse(y_pred, sparse_val=79), dtype="int64")

        # y_true_sparse = K.cast(K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32')), dtype="int64")
        y_true_sparse = K.cast(dense_to_sparse(y_true, sparse_val=79), dtype="int64")

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(tf.cast(len(y_true), dtype="float32"))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_states(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


class WERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Word Error Rate
    """

    def __init__(self, name='WER_metric', **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="wer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        #
        # decode, log = K.ctc_decode(y_pred,
        #                            input_length,
        #                            greedy=False, beam_width=16)
        # decode = tf.transpose(decode, [1, 0])
        # decode = K.ctc_label_dense_to_sparse(decode, K.cast(input_length, 'int32'))
        decode = K.cast(dense_to_sparse(y_pred, sparse_val=79), dtype="int64")

        # y_true_sparse = K.cast(K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32')), dtype="int64")
        y_true_sparse = K.cast(dense_to_sparse(y_true, sparse_val=79), dtype="int64")

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))
        # if correct_words_amount.numpy() < 6.0:
        #     print(decode)
        #     print(y_true)
        #     print(y_true_sparse)
        #     print(correct_words_amount.numpy())
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        self.wer_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(tf.cast(len(y_true), dtype="float32"))
        # self.counter.assign_add(len(y_true))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_states(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)

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
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm1")(x)

    x = TimeDistributed(layers.Conv2D(16, 3, padding="valid"), name="conv2")(x)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = TimeDistributed(layers.MaxPooling2D(), name="pool2")(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm2")(x)

    x = TimeDistributed(layers.Conv2D(32, 3, padding="valid"), name="conv3")(x)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm3")(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop1")(x)

    x = TimeDistributed(layers.Conv2D(64, 3, padding="valid"), name="conv4")(x)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm4")(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop2")(x)

    x = TimeDistributed(layers.Conv2D(128, 3, padding="valid"), name="conv5")(x)
    x = TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
    x = TimeDistributed(layers.BatchNormalization(), name="batch_norm5")(x)
    x = TimeDistributed(layers.Dropout(0.2), name="drop3")(x)
    x = TimeDistributed(layers.Flatten(), name="flatten")(x)
    out = TimeDistributed(layers.Dense(256, activation="relu"))(x)
    return out


def rnn(inputs, charset_len):
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(charset_len + 1, activation='softmax')(x)
    return out

class OcrModel(object):
    def __init__(self, charset_len, batch_size, frame_size):
        inputs = layers.Input(shape=[None, frame_size, frame_size, 1], name="images", dtype="float32", batch_size=batch_size)
        labels = layers.Input(shape=[None], dtype="int32", name="labels", batch_size=batch_size)

        cnn_out = cnn(inputs)

        rnn_out = rnn(cnn_out, charset_len)
        output = CTCLayer(name="ctc_loss")(labels, rnn_out)

        opt = tf.keras.optimizers.Adam(lr=0.0003)
        self.model = tf.keras.models.Model(inputs={"images": inputs, "labels": labels}, outputs=[output], name="ocr_nn_v1")
        print(self.model.summary(line_length=150))
        cer = CERMetric()
        wer = WERMetric(name="LER_metric")
        self.model.compile(opt, metrics=[cer, wer])
