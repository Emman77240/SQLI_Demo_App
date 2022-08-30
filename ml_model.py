import nltk
import pickle
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Sequential, load_model
from data_sanitize import clean_data
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer



# download stopwords
#nltk.download('stopwords')


# read csv data
df = pd.read_csv('sqli.csv', encoding='utf-16')

# vectorize the data
vectorizer = CountVectorizer(min_df = 2, max_df = 0.7, max_features = 4096, stop_words = nltk.corpus.stopwords.words('english'))
posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()
print(posts.shape)
posts.shape = (4200, 64, 64, 1)
X = posts
y = df['Label']


# split test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
Xtrain = X_train.copy()
Xtrain.shape = (X_train.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
Xtest = X_test.copy()
Xtest.shape = (Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])

# specify classifier 
NB_clf = GaussianNB()
NB_clf.fit(Xtrain, y_train)

# run prediction for Naive Bayes
NB_pred = NB_clf.predict(Xtest)

print(f"Test dataset accuracy with Naive Bayes: {accuracy_score(NB_pred, y_test)}")

model = Sequential([
    layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(64,64,1)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128,(3,3), activation=tf.nn.relu),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(256,(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs = 10, verbose = True, validation_data = (X_test, y_test), batch_size = 3)

# run prediction for CNN
prediction = model.predict(X_test)

for i in range(len(prediction)):
    if prediction[i] > 0.5:
        prediction[i] = 1
    elif prediction[i] <= 0.5:
        prediction[i] = 0

print(f"Test dataset accuracy with CNN: {accuracy_score(y_test, prediction)}")

# save CNN model and vectorizer
model.save('my_CNN_model.h5')
with open('CNN_vectorizer', 'wb') as fin:
    pickle.dump(vectorizer, fin)

