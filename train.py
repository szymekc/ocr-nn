import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.run_functions_eagerly(True)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
from model import OcrModel
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from callbacks import SavedHistory, ToggleMetrics
from dataset import train_gen, val_gen, frame_h, frame_w, characters


batch_size = 32
learning_rate = 0.0001

train_batches = tf.data.Dataset.from_generator(
    train_gen,
    output_types=({"images": tf.float32, "labels": tf.float32}, tf.float32),
).padded_batch(
    batch_size,
    padded_shapes=({'images': [None, frame_h, frame_w, 1], 'labels': [None]}, [None]),
    padding_values=79.
)
val_batches = tf.data.Dataset.from_generator(
    val_gen,
    output_types=({"images": tf.float32, "labels": tf.float32}, tf.float32),
).padded_batch(
    batch_size,
    padded_shapes=({'images': [None, frame_h, frame_w, 1], 'labels': [None]}, [None]),
    padding_values=79.
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)
history = SavedHistory()
toggle_metrics = ToggleMetrics()
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
checkpoint = ModelCheckpoint("best_model_rect_64_lrad.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
model = OcrModel(len(characters), batch_size, frame_h, frame_w, learning_rate)
model.model.load_weights("best_model_rect_64.hdf5")
training = model.model.fit(train_batches, validation_data=val_batches, batch_size=batch_size, epochs=1000, callbacks=[checkpoint, early_stopping, history, rlrop], verbose=1, shuffle=True)
