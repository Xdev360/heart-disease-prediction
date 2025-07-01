from flask import Flask
from .routes import create_routes

def create_app():
    app = Flask(__name__)
    app.secret_key = 'your_secret_key_here'
    create_routes(app)
    return app 