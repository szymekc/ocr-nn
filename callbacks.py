import tensorflow as tf
import pickle


class SavedHistory(tf.keras.callbacks.History):
    def __init__(self):
        super(SavedHistory, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        super(SavedHistory, self).on_epoch_end(epoch, logs)
        with open("history.pkl", "wb") as f:
            pickle.dump(self.history, f)


class ToggleMetrics(tf.keras.callbacks.Callback):
    '''On test begin (i.e. when evaluate() is called or
     validation data is run during fit()) toggle metric flag '''
    def on_test_begin(self, logs=None):
        for metric in self.model.metrics:
            if 'ER' in metric.name:
                metric.on.assign(True)

    def on_test_end(self,  logs=None):
        for metric in self.model.metrics:
            if 'ER' in metric.name:
                metric.on.assign(False)