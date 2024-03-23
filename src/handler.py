import os
import random
import torch
import io
import base64
import numpy as np
import soundfile as sf
import runpod
from runpod.serverless.utils import rp_download
from runpod.serverless.utils.rp_validator import validate
from audiosr import build_model, super_resolution
from rp_schemas import INPUT_SCHEMA

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

# Assume this is the sample rate your model outputs. Adjust accordingly.
FIXED_SAMPLE_RATE = 48000

builtModelSr = build_model(model_name='basic', device='auto')

def handler(job):
    job_input = job['input']
    validated_input = validate(job_input, INPUT_SCHEMA)
    
    if 'errors' in validated_input:
        return {"errors": validated_input['errors']}

    validated_input = validated_input['validated_input']
    input_file_url = validated_input['input_file_url']

    downloaded_file = rp_download.file(input_file_url)
    if not downloaded_file.get("success", True):
        error_msg = f"Failed to download the input file. URL: {input_file_url}"
        return {"errors": error_msg}

    input_file_path = downloaded_file["file_path"]
    seed = validated_input['seed'] or random.randint(0, 2**32 - 1)
    ddim_steps = validated_input['ddim_steps']
    guidance_scale = validated_input['guidance_scale']

    # Adjusted to expect only waveform since sr (sample rate) is known/fixed
    waveform = super_resolution(
        builtModelSr,
        input_file=input_file_path,
        seed=seed,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps
    )

    # Use the FIXED_SAMPLE_RATE for the sample rate
    out_wav = (waveform[0] * 32767).astype(np.int16).T
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, data=out_wav, samplerate=FIXED_SAMPLE_RATE, format='WAV')
    audio_bytes.seek(0)

    # Encode the WAV file to a base64 string
    base64_audio = base64.b64encode(audio_bytes.read()).decode('utf-8')

    # Return the base64 encoded audio with the appropriate data URI scheme
    return f"data:audio/wav;base64,{base64_audio}"

# Assuming runpod.serverless.start correctly defined elsewhere
runpod.serverless.start({"handler": handler})
