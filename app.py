from flask import Flask, Response
from scripts.txt2img_flask import *
app = Flask(__name__)

@app.route('/')
def index():
    img = generate("an astronaut riding on a horse, artstation")
    return Response(img, mimetype="image/png")
