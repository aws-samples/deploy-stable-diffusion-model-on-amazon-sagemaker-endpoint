from diffusers import StableDiffusionInpaintPipeline
import torch
import base64
import numpy as np
from PIL import Image

height, width = 512, 512


def process_data(data: dict) -> dict:
    global height, width
    height = data.pop("height", 512)
    width = data.pop("width", 512)

    init_image_decoded = np.reshape(
        np.frombuffer(
            base64.decodebytes(bytes(data.pop("image"), encoding="utf-8")),
            dtype=np.uint8,
        ),
        (height, width, 3),
    )

    mask_image_decoded = np.reshape(
        np.frombuffer(
            base64.decodebytes(bytes(data.pop("mask_image"), encoding="utf-8")),
            dtype=np.uint8,
        ),
        (height, width, 3),
    )

    return {
        "prompt": data.pop("prompt", data),
        "image": Image.fromarray(init_image_decoded),
        "mask_image": Image.fromarray(mask_image_decoded),
        "strength": data.pop("strength", 0.75),
        "guidance_scale": data.pop("guidance_scale", 7.5),
        "num_inference_steps": min(data.pop("num_inference_steps", 50), 50),
        "height": height,
        "width": width,
    }


def model_fn(model_dir: str):
    inp_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
    )
    if torch.cuda.is_available():
        inp_pipe = inp_pipe.to("cuda")

    inp_pipe.enable_attention_slicing()
    return inp_pipe


def predict_fn(data: dict, hgf_pipe) -> dict:

    with torch.autocast("cuda"):
        images = hgf_pipe(**process_data(data))["images"]

    # return dictionary, which will be json serializable
    return {
        "images": [
            base64.b64encode(np.array(image).astype(np.uint8)).decode("utf-8")
            for image in images
        ],
        "height": height,
        "width": width,
    }
