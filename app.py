from flask import Flask, Response, render_template, request
from scripts.txt2img_flask import *
from base64 import b64encode
from urllib.parse import quote

app = Flask(__name__)

@app.route('/txt2img')
def page_txt2img():
    return render_template("generate.html")

@app.route('/txt2img/endpoint', methods=['POST'])
def page_txt2img_endpoint():

    input = request.get_json()

    print(request.data)

    print("Generating: "+input["prompt"])

    img = generate(input["prompt"])

    img_data = b64encode(img.getvalue()).decode('ascii')
    return 'data:image/png;base64,' + quote(img_data)


    # Response(img, mimetype="image/png")
