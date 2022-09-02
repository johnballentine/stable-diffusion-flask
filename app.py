from flask import Flask, Response, render_template, request
from scripts.txt2img_flask import *
from base64 import b64encode
from urllib.parse import quote
from dotenv import load_dotenv

# Local imports
from models import *

app = Flask(__name__)
load_dotenv()

database_url = os.getenv('DATABASE_URL')
if database_url[:11] == "postgres://":
    # Fix URL for flask
    database_url = database_url[:8] + "ql" + database_url[8:]

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def save_generation(opt, image_data, request_headers="", request_raw=""):
    generation = Generation(
        opt=opt,
        image_data=image_data,
        request_raw=request_raw,
    )
    db.session.add(generation)
    db.session.commit()

    return generation


@app.route('/generate')
def page_generate():
    return render_template("generate.html")

@app.route('/txt2img/endpoint', methods=['POST'])
def page_txt2img_endpoint():

    input = request.get_json()

    opt = {
        "prompt": input["prompt"],
        "outdir": "outputs/txt2img-samples",
        "skip_grid": None,
        "skip_save": None,
        "ddim_steps": 50,
        "plms": True,
        "laion400m": None,
        "fixed_code": None,
        "ddim_eta": 0.0,
        "n_iter": 1,
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "n_samples": 1,
        "n_rows": 0,
        "scale": 7.5,
        "from_file": None,
        "config": "configs/stable-diffusion/v1-inference.yaml",
        "ckpt": "models/ldm/stable-diffusion-v1/model.ckpt",
        "seed": 42,
        "precision": "autocast"
    }

    print("Generating: "+opt["prompt"])

    image = generate(opt)
    image_data = 'data:image/png;base64,' + quote(b64encode(image.getvalue()).decode('ascii'))

    save_generation(opt=opt,
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
