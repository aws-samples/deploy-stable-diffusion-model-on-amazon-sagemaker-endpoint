# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import base64
import json
import sys
import io, os
import signal
import traceback
import flask

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

apply_hed = HEDdetector()
model_dir = "/opt/ml/model"

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded
    
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = create_model(f'{model_dir}/cldm_v15.yaml').cpu()
            cls.model.load_state_dict(load_state_dict(f'{model_dir}/control_sd15_hed.pth', location='cuda'))
            cls.model = cls.model.cuda()
        return cls.model
    
    @classmethod
    def predict_fn(cls, data):
#         body = json.loads(data)
        body = data
        H = body['H'] 
        W = body['W'] 
        C = body['C']

        detected_map = np.reshape(
            np.frombuffer(
                base64.decodebytes(bytes(body['detected_map'], encoding="utf-8")),
                dtype=np.uint8,
            ),
            (H, W, C),
        )

        prompt = body['prompt'] 
        a_prompt = body['a_prompt']
        n_prompt = body['n_prompt'] 
        num_samples = body['num_samples']  
        ddim_steps = body['ddim_steps']  
        guess_mode = body['guess_mode']  
        strength = body['strength']  
        scale = body['scale']  
        seed = body['seed']  
        eta = body['eta']  


        cls.model = cls.get_model()
        ddim_sampler = DDIMSampler(cls.model)
        with torch.no_grad():
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                cls.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [cls.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [cls.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                cls.model.low_vram_shift(is_diffusing=True)

            cls.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                         shape, cond, verbose=False, eta=eta,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond)

            if config.save_memory:
                cls.model.low_vram_shift(is_diffusing=False)

            x_samples = cls.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = {"detected_map": detected_map.tolist(),
               "image": x_samples.tolist()
              }
        return results


# The flask app for serving predictions
app = flask.Flask(__name__)

ScoringService.get_model()

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    body = None
    try:
        if flask.request.content_type == "application/json":
            data = flask.request.get_json()
        else:
            return flask.Response(
                response="This predictor only supports application/json data", status=415, mimetype="text/plain"
            )
    

        # Do the prediction
        predictions = ScoringService.predict_fn(data)

        return flask.Response(response=json.dumps(predictions), status=200, mimetype="application/json")
    except Exception as e:
        print(str(e))
        result = {"error": f"Internal server error"}
        return flask.Response(response=result, status=500, mimetype="application/json")


