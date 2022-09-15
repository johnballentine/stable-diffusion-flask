txt2img_opt_defaults = {
    "prompt": "an astronaut riding a horse",
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