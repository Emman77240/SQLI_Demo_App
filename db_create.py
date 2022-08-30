from multiprocessing.util import LOGGER_NAME
from app import db
from models import BlogPost


# create the database and db tables
db.create_all()

# insert db items
db.session.add(BlogPost("Good", "I\'m good.")) 
db.session.add(BlogPost("Well", "I\'m well."))


# commit changes
db.session.commit()