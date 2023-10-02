import os
from flask import Flask

# init the app for flask
app = Flask(__name__,
            template_folder=os.getcwd() + '/view/templates/',
            static_folder=os.getcwd() + '/view/static',
            static_url_path=os.getcwd() + '/view/static')

