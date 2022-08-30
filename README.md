# SQLI Demo App

SQLI Demo App is a Python web application built with flask which demonstrates user input validation using machine learning techniques.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install app requirements.

```bash
pip install -r requirements.txt
```

## Usage

```python
import pickle
from keras.models import load_model


# Load ml model and vectorizer
ml_model = load_model('my_CNN_model.h5')
ml_vectorizer = pickle.load(open("CNN_vectorizer", "rb"))

# Grab login input for SQLI prediction
ml_text = clean_data(request.form['username'] + ' ' + request.form['password'])
ml_text = [ml_text]
ml_text = ml_vectorizer.transform(ml_text).toarray()
ml_text.shape = (1, 64, 64, 1)
result = ml_model.predict(ml_text)
```


## Outcomes
Result in the above code snippet must be analyzed as it indicates the presence of SQL injection attack via login input.