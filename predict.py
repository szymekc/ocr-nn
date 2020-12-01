import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.run_functions_eagerly(True)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
from test_display import show_image
from preprocess import binarize, resize, make_frames, erode, binarize_adaptive, denoise
from model import OcrModel, word_beam_search_module, corpus, words, word_chars
from dataset import test_gen, frame_h, frame_w, characters, num_to_char
from segment import par_to_batch
import termcolor
frames_num = 128
batch_size = 32
image_path = "./paratest.png"


def pred_gen():
    for img in par_to_batch(image_path):
        yield tf.reshape(predict_prep(img), (1, frames_num, frame_h, frame_w, 1))


def predict_prep(image):
    image = erode(image)
    image = binarize(image, 100)
    image = denoise(image)
    image = resize(image, frame_h)
    image = make_frames(image, frame_h, frame_w, frames_num)
    return image


def decode_pred(pred):
    y_pred = tf.transpose(pred, [1, 0, 2])
    y_pred = word_beam_search_module.word_beam_search(y_pred, 25, 'Words', 0, corpus, words, word_chars)
    string = tf.strings.reduce_join(num_to_char(tf.cast(y_pred, dtype="int64"))).numpy()
    return str(string.decode("utf-8")).replace("[UNK]", "").replace("|", " ")


model = OcrModel(len(characters), batch_size, frame_h, frame_w)
prediction_model = tf.keras.models.Model(
    model.model.get_layer(name="images").input, model.model.get_layer(name="dense_2").output
)
prediction_model.summary()
prediction_model.load_weights("best_model_rect_64.hdf5")
# image = tf.reshape(predict_prep("./pred5.png"), (batch_size, frames_num, frame_h, frame_w, 1))
pred = prediction_model.predict(pred_gen(), batch_size=batch_size)
print(termcolor.colored(decode_pred(pred).replace("""\"""", "\n"), "yellow"))