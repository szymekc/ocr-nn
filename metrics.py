import tensorflow as tf
import tensorflow.keras.backend as K


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


def tf_count_words(t):
    elements_equal_to_value = tf.equal(t, 78)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    sub = tf.fill(count.shape, 1)
    return tf.add(count, sub)


class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    https://stackoverflow.com/questions/60285167/tensorflow-callback-as-custom-metric-for-ctc
    """

    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")
        self.on = tf.Variable(True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.on:
            decode = K.cast(dense_to_sparse(y_pred, sparse_val=79), dtype="int64")

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
    A custom Keras metric to compute the Line Error Rate
    https://stackoverflow.com/questions/60285167/tensorflow-callback-as-custom-metric-for-ctc
    """

    def __init__(self, name='WER_metric', **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(name="total_ler", initializer="zeros")
        self.counter = self.add_weight(name="ler_count", initializer="zeros")
        self.on = tf.Variable(True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.on:

            decode = K.cast(dense_to_sparse(y_pred, sparse_val=79), dtype="int64")
            word_counts = tf.map_fn(tf_count_words, y_true, fn_output_signature=tf.int32)
            y_true_sparse = K.cast(dense_to_sparse(y_true, sparse_val=79), dtype="int64")

            decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
            distance = tf.edit_distance(decode, y_true_sparse, normalize=False)
            correct_words_amount = tf.reduce_sum(tf.divide(distance, tf.cast(word_counts, dtype="float32")))
            self.wer_accumulator.assign_add(correct_words_amount)
            self.counter.assign_add(tf.cast(len(y_true), dtype="float32"))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_states(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)



class LERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Line Error Rate
    https://stackoverflow.com/questions/60285167/tensorflow-callback-as-custom-metric-for-ctc
    """

    def __init__(self, name='LER_metric', **kwargs):
        super(LERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(name="total_ler", initializer="zeros")
        self.counter = self.add_weight(name="ler_count", initializer="zeros")
        self.on = tf.Variable(True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.on:
            decode = K.cast(dense_to_sparse(y_pred, sparse_val=79), dtype="int64")

            y_true_sparse = K.cast(dense_to_sparse(y_true, sparse_val=79), dtype="int64")

            decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
            distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
            correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))
            self.wer_accumulator.assign_add(correct_words_amount)
            self.counter.assign_add(tf.cast(len(y_true), dtype="float32"))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_states(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)
