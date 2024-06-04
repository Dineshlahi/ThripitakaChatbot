# app/__init__.py

from flask import Flask
from .chatbot import chat_bot

def create_app():
    # Create the Flask app
    app = Flask(__name__)

    # Load configuration (if any)
    app.config.from_pyfile('config.py', silent=True)

    # Register the chatbot Blueprint
    app.register_blueprint(chat_bot, url_prefix='/chatbot')

    # You can add more Blueprints for other features if needed

    return app

