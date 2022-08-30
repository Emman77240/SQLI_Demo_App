# initialize sql alchemy database url
uri = 'sqlite:///posts.db'

# default config
class BaseConfig(object):
    DEBUG = False
    SECRET_KEY = ''
    SQLALCHEMY_DATABASE_URI = uri
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False