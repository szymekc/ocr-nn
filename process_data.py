from preprocess import preprocess_all_data, split_dataset
import pickle
images = pickle.load(open("images.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))

# preprocessed_x_train, preprocessed_y_train, preprocessed_x_val, preprocessed_y_val, preprocessed_x_test, preprocessed_y_test = split_dataset(*preprocess_all_data(images, labels))
preprocessed_x_train = pickle.load(open("preprocessed_x_train.pkl", "rb"))
preprocessed_y_train = pickle.load(open("preprocessed_y_train.pkl", "rb"))
preprocessed_x_val = pickle.load(open("preprocessed_x_val.pkl", "rb"))
preprocessed_y_val = pickle.load(open("preprocessed_y_val.pkl", "rb"))
preprocessed_x_test = pickle.load(open("preprocessed_x_test.pkl", "rb"))
preprocessed_y_test = pickle.load(open("preprocessed_y_test.pkl", "rb"))

dataset = [preprocessed_x_train, preprocessed_x_val, preprocessed_x_test]


pickle.dump(preprocessed_x_train, open('preprocessed_x_train.pkl', 'wb'))
pickle.dump(preprocessed_y_train, open('preprocessed_y_train.pkl', 'wb'))
pickle.dump(preprocessed_x_val, open('preprocessed_x_val.pkl', 'wb'))
pickle.dump(preprocessed_y_val, open('preprocessed_y_val.pkl', 'wb'))
pickle.dump(preprocessed_x_test, open('preprocessed_x_test.pkl', 'wb'))
pickle.dump(preprocessed_y_test, open('preprocessed_y_test.pkl', 'wb'))
