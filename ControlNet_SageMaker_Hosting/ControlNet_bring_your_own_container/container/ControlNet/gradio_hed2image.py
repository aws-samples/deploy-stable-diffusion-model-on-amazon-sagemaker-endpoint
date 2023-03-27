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
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


import sagemaker
import boto3

region = os.environ['region']
boto_session = boto3.Session(region_name=region)
sagemaker_client = boto_session.client("sagemaker")
sess = sagemaker.session.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client
)
role = sess.get_caller_identity_arn()
sm_runtime = boto3.client("sagemaker-runtime", region_name=region)


apply_hed = HEDdetector()


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    
    input_image = HWC3(input_image)
    detected_map = apply_hed(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR) 
    
    data = {
        "detected_map": base64.b64encode(detected_map).decode("utf-8"),
        "prompt": prompt, 
        "a_prompt": a_prompt, 
        "n_prompt": n_prompt, 
        "num_samples": num_samples,  
        "ddim_steps": ddim_steps, 
        "guess_mode": guess_mode, 
        "strength": strength, 
        "scale": scale, 
        "seed": seed, 
        "eta": eta, 
        "H": H, 
        "W": W, 
        "C": C
           }

    endpoint_name = os.environ['endpoint_name']
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(data),
        ContentType="application/json",
    )
    x_samples = response["Body"].read().decode('utf-8')
    output = json.loads(x_samples)
    image = output["image"]
    image = np.array(image)

    results = [image[i] for i in range(num_samples)]
    return [detected_map] + results



block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with HED Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="HED Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
#             
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    

block.launch(server_name='0.0.0.0',share=True)

