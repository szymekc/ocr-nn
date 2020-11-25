from model import OcrModel
import pickle
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import dataset
import itertools
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# Mapping characters to integers
from preprocess import preprocess_sample_train, preprocess_sample_test, characters, frame_size, sample_train, sample_test, sample_val


batch_size = 6

train_batches = tf.data.Dataset.from_generator(
    sample_train,
    output_types=({"images": tf.float32, "labels": tf.float32}, tf.float32),
).padded_batch(
    batch_size,
    padded_shapes=({'images': [None, frame_size, frame_size, 1], 'labels': [None]}, [None]),
    padding_values=79.
)
val_batches = tf.data.Dataset.from_generator(
    sample_val,
    output_types=({"images": tf.float32, "labels": tf.float32}, tf.float32),
).padded_batch(
    batch_size,
    padded_shapes=({'images': [None, frame_size, frame_size, 1], 'labels': [None]}, [None]),
    padding_values=79.
)
test_batches = tf.data.Dataset.from_generator(
    sample_test,
    output_types=({"images": tf.float32, "labels": tf.float32}, tf.float32),
).padded_batch(
    batch_size,
    padded_shapes=({'images': [None, frame_size, frame_size, 1], 'labels': [None]}, [None]),
    padding_values=79.
)

# _, ax = plt.subplots(8, 8, figsize=(20, 10))
# for batch in train_batches.take(1):
#     images = batch[0]
#     labels = batch[1]
#     label = tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode("utf-8")
#     for i in range(64):
#         img = (images[1] * 255).numpy().astype("uint8")
#         ax[i // 8, i % 8].imshow(img[i*3, :, :, 0].T, cmap="gray")
#         ax[i // 8, i % 8].set_title(label)
#         ax[i // 8, i % 8].axis("off")
# plt.show()

early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

model = OcrModel(len(characters), batch_size, frame_size)
# model.model.load_weights("weights.h5")
history = model.model.fit(train_batches, validation_data=val_batches, batch_size=batch_size, epochs=50, callbacks=[early_stopping])
model.model.save_weights("weights.h5")
# print(model.model.evaluate(test_batches, batch_size=batch_size, callbacks=[early_stopping]))
#results = model.model.evaluate(x=[x_test, y_test, x_test_len, y_test_len], batch_size=32)
print('dupa')
