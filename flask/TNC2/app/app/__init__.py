from flask import Flask
from flask_dropzone import Dropzone

app = Flask(__name__)
dropzone = Dropzone(app)

from app import model_server_test
from app import web_server_test

