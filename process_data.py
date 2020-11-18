from preprocess import preprocess_all
import pickle


preprocessed_dataset = preprocess_all(data[0])

pickle.dump(preprocessed_dataset, open('data_preprocessed.pkl', 'wb'))
pickle.dump(data[1], open('labels.pkl', 'wb'))
