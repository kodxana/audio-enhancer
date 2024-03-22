""" Example handler file. """
import random
import os
import runpod
import base64
from audiosr import build_model, super_resolution
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

builtModelSr = build_model(model_name='basic', device='auto')

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

def handler(job):
    job_input = job['input']
    if not job_input.input_file:
        return "No Input File provided."

    input_file = job_input.input_file
    ddim_steps = job_input.get('ddim_steps', 50)
    seed = job_input.get('seed', 42)
    guidance_scale = job_input.get('guidance_scale', 3.5)

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"Setting seed to: {seed}")

    waveform = super_resolution(
        builtModelSr,
        input_file,
        seed=seed,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
        latent_t_per_second=12.8
    )

    return waveform

# Start serverless with the defined handler.
runpod.serverless.start({"handler": handler})