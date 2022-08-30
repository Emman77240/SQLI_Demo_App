import pickle
from data_sanitize import clean_data
from keras.models import load_model



text = "admin"
text = clean_data(text)
text = [text]



ml_model = load_model('my_CNN_model.h5')
ml_vectorizer = pickle.load(open("CNN_vectorizer", "rb"))

text = ml_vectorizer.transform(text).toarray()
text.shape = (1, 64, 64, 1)
result = ml_model.predict(text)

print(result > 0.5)
