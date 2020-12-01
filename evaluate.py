import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.run_functions_eagerly(True)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
from model import OcrModel
from keras.callbacks import ModelCheckpoint
from callbacks import SavedHistory, ToggleMetrics
from dataset import test_gen, frame_h, frame_w, characters

batch_size = 32

test_batches = tf.data.Dataset.from_generator(
    test_gen,
    output_types=({"images": tf.float32, "labels": tf.float32}, tf.float32),
).padded_batch(
    batch_size,
    padded_shapes=({'images': [None, frame_h, frame_w, 1], 'labels': [None]}, [None]),
    padding_values=79.
)

model = OcrModel(len(characters), batch_size, frame_h, frame_w)
model.model.load_weights("best_model_rect_64.hdf5")
toggle_metrics = ToggleMetrics()
print(model.model.evaluate(test_batches, batch_size=batch_size, callbacks=[toggle_metrics]))
