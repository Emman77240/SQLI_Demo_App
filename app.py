import os
import pickle
from functools import wraps
from keras.models import load_model
from data_sanitize import clean_data
from flask_sqlalchemy import SQLAlchemy 
from flask import Flask, render_template, g, flash, redirect, url_for, request, session



# create the application object
app = Flask(__name__)


# configure app settings
app.config.from_object(os.environ['APP_SETTINGS'])



# create the sqlalchemy object
db = SQLAlchemy(app)



# load ml model and vectorizer
ml_model = load_model('my_CNN_model.h5')
ml_vectorizer = pickle.load(open("CNN_vectorizer", "rb"))




from models import *

# login required decorator
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap


# link the functions to a url with decorators
@app.route('/')
@login_required
def home():      

    # query database for posts
    posts = db.session.query(BlogPost).all()


    # render a template
    return render_template('index.html', posts=posts)


@app.route('/welcome')
def welcome():
    return render_template("welcome.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    # initialize error
    error = None

    if request.method == 'POST':
        """Assign variable names to form entries"""
        login_text = str(request.form['username'])
        password_text = str(request.form['password'])

        """Check if the user is admin"""
        if login_text == 'admin' and password_text == 'admin':
            session['logged_in'] = True
            flash('You were just logged in!')
            return redirect(url_for('home'))
        else:
            """Grab login input for SQLI prediction"""
            ml_text = clean_data(login_text + ' ' + password_text)
            ml_text = [ml_text]
            ml_text = ml_vectorizer.transform(ml_text).toarray()
            ml_text.shape = (1, 64, 64, 1)
            result = ml_model.predict(ml_text)

            if (result > 0.5) == True:
                error = 'SQL injection detected!'
            
            else:
                error = 'Invalid credentials. Please try again.'
            
        


    return render_template('login.html', error = error)


@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    flash('You were just logged out!')
    return redirect(url_for('welcome'))


#def connect_db():
#    return sqlite3.connect('posts.db')




if __name__ == '__main__':
    app.run()