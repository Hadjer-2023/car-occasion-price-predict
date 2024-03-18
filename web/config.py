import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    APP_NAME = "VoiturEstim"
    APP_SPEECH = "Estimer facilement le prix d'une voiture"
    SECRET_KEY = "you_will_never_guess_even_with_brute_force"
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir, "app.db")