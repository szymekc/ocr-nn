from model import OcrModel
import pickle
import tensorflow as tf
import numpy as np
import dataset
import itertools
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# Mapping characters to integers
from preprocess import x_train, y_train, x_test, y_test, preprocess_sample_train, preprocess_sample_test, characters

batch_size = 8
num_train = len(x_train)
num_test = len(x_test)

train_batches = tf.data.Dataset.from_generator(
    preprocess_sample_train,
    output_types=({"images": tf.float32, "labels": tf.int64}, tf.int64),
).padded_batch(
    batch_size,
    padded_shapes=({'images': [None, 128, 128, 1], 'labels': [None]}, [None])
)
test_batches = tf.data.Dataset.from_generator(
    preprocess_sample_test,
    output_types=({"images": tf.float32, "labels": tf.int64}, tf.int64),
).padded_batch(
    batch_size,
    padded_shapes=({'images': [None, 128, 128, 1], 'labels': [None]}, [None])
)

# _, ax = plt.subplots(8, 8, figsize=(20, 10))
# for batch in x_train_batches.take(1):
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

model = OcrModel(len(characters),batch_size)
history = model.model.fit(train_batches, validation_data=test_batches, batch_size=batch_size, epochs=100, callbacks=[early_stopping])
#results = model.model.evaluate(x=[x_test, y_test, x_test_len, y_test_len], batch_size=32)
print('dupa')
