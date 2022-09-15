from flask import Flask, Response, render_template, request
from base64 import b64encode
from urllib.parse import quote
from dotenv import load_dotenv
from inspect import getfullargspec
import os

# Local imports
from models import db, Generation
from scripts.txt2img_flask import Txt2Img

app = Flask(__name__)
load_dotenv()

database_url = os.getenv('DATABASE_URL')
if database_url[:11] == "postgres://":
    # Fix URL for flask
    database_url = database_url[:8] + "ql" + database_url[8:]

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def get_params(func, exclude_keys=["self"]):
    spec = getfullargspec(func)
    padded_defaults = (None,) * (len(spec.args) - len(spec.defaults)) + spec.defaults

    params = dict(zip(spec.args, padded_defaults))
    for key in exclude_keys:
        params.pop(key)

    return params

# Used for table rows/columns
def chunk_params(opt, chunks=2):
    list_temp = [opt] * chunks
    opt_chunked = []
    for i, item in enumerate(list_temp):
        opt_chunked.append(opt[i:i+chunks])

    return opt_chunked
    

def save_generation(opt, image_data, request_headers="", request_raw=""):
    generation = Generation(
        opt=opt,
        image_data=image_data,
        request_raw=request_raw,
    )
    db.session.add(generation)
    db.session.commit()

    return generation

@app.route('/')
def index():
    return "Running..."


@app.route('/generate')
def page_generate():

    excludes = ['self', 'prompt', 'outdir', 'laion400m',
                'from_file', 'config', 'ckpt']

    opt = get_params(Txt2Img.__init__, excludes)

    opt_chunked = chunk_params(opt)

    print(type(opt_chunked))

    for item in opt_chunked:
        print(str(item))
    
    return render_template("generate.html", opt_chunked=opt_chunked)


@app.route('/txt2img/endpoint', methods=['POST'])
def page_txt2img_endpoint():

    input = request.get_json()
    """opt = dict(txt2img_opt_defaults)

    opt["prompt"] = input["prompt"]"""

    print("Generating: "+input["prompt"])

    generation = Txt2Img(input["prompt"])

    image = generation.generate()
    image_data = 'data:image/png;base64,' + quote(b64encode(image.getvalue()).decode('ascii'))

    save_generation(opt={"test":"test"},
                    image_data=image_data,
                    request_headers=str(request.headers),
                    request_raw=request.get_data(as_text=True))

    return image_data


@app.route('/gallery')
def gallery():
    generations = (Generation.query
                  .order_by(Generation.id.desc())
                  .limit(10)
                  )

    return render_template("gallery.html", generations=generations)


@app.route('/output')
def output():
    max_images = 100
    filenames = os.listdir(os.path.join(app.static_folder, 'outputs/samples'))
    filenames = sorted(filenames, reverse=True)
    filenames = filenames[0:max_images]

    return render_template("output.html", filenames=filenames)


if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0')
