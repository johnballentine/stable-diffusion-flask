import os, sys, glob  # , argparse
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

# imports for flask version
import io


class Txt2Img():
    def __init__(self, prompt="an astronaut riding a horse",
                 outdir="outputs/txt2img-samples",
                 skip_grid=None, skip_save=None, ddim_steps=50, plms=True,
                 laion400m=None, fixed_code=None, ddim_eta=0.0, n_iter=1,
                 H=512, W=512, C=4, f=8, n_samples=1, n_rows=0, scale=7.5,
                 from_file=None, config="configs/stable-diffusion/v1-inference.yaml",
                 ckpt="models/ldm/stable-diffusion-v1/model.ckpt", seed=42,
                 precision="autocast"):

        self.prompt=prompt
        self.outdir=outdir
        self.skip_grid=skip_grid
        self.skip_save=skip_save
        self.ddim_steps=ddim_steps
        self.plms=plms
        self.laion400m=laion400m
        self.fixed_code=fixed_code
        self.ddim_eta=ddim_eta
        self.n_iter=n_iter
        self.H=H
        self.W=W
        self.C=C
        self.f=f
        self.n_samples=n_samples
        self.n_rows=n_rows
        self.scale=scale
        self.from_file=from_file
        self.config=config
        self.ckpt=ckpt
        self.seed=seed
        self.precision=precision

        # load safety model
        self.safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(self.safety_model_id)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(self.safety_model_id)

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())


    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model


    def put_watermark(self, img, wm_encoder=None):
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img


    def load_replacement(self, x):
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y)/255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x


    def check_safety(self, x_image):
        safety_checker_input = self.safety_feature_extractor(self.numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = self.load_replacement(x_checked_image[i])
        return x_checked_image, has_nsfw_concept


    def generate(self):
        if self.laion400m:
            print("Falling back to LAION 400M model...")
            self.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            self.ckpt = "models/ldm/text2img-large/model.ckpt"
            self.outdir = "outputs/txt2img-samples-laion400m"

        seed_everything(self.seed)

        config = OmegaConf.load(f"{self.config}")
        model = self.load_model_from_config(config, f"{self.ckpt}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if self.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(self.outdir, exist_ok=True)
        outpath = self.outdir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        batch_size = self.n_samples
        n_rows = self.n_rows if self.n_rows > 0 else batch_size
        if not self.from_file:
            assert self.prompt is not None
            data = [batch_size * [self.prompt]]

        else:
            print(f"reading prompts from {self.from_file}")
            with open(self.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(self.chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if self.fixed_code:
            start_code = torch.randn([self.n_samples, self.C, self.H // self.f, self.W // self.f], device=device)

        precision_scope = autocast if self.precision=="autocast" else nullcontext

        # Set up memory file
        imgbuf = io.BytesIO()

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if self.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [self.C, self.H // self.f, self.W // self.f]
                            samples_ddim, _ = sampler.sample(S=self.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=self.n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=self.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=self.ddim_eta,
                                                             x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = self.check_safety(x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not self.skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img = self.put_watermark(img, wm_encoder)
                                    img.save(imgbuf, format='PNG')
                                    base_count += 1

                            if not self.skip_grid:
                                all_samples.append(x_checked_image_torch)

                    if not self.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img = self.put_watermark(img, wm_encoder)
                        img.save(imgbuf, format='PNG')
                        grid_count += 1

                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
              f" \nEnjoy.")

        imgbuf.seek(0)  # reset buffer position in memory

        return imgbuf
