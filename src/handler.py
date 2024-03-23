# handler.py

import os
import random
import torch
from runpod.serverless.utils import rp_download, rp_upload
from runpod.serverless.utils.rp_validator import validate
from audiosr import build_model, super_resolution
from rp_schemas import INPUT_SCHEMA


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

builtModelSr = build_model(model_name='basic', device='auto')

def handler(job):
    job_input = job['input']
    validated_input = validate(job_input, INPUT_SCHEMA)
    
    if 'errors' in validated_input:
        return {"errors": validated_input['errors']}
    
    validated_input = validated_input['validated_input']
    input_file_url = validated_input['input_file_url']
    
    downloaded_file = rp_download.file(input_file_url)
    if downloaded_file["success"] is not True:
        return {"errors": "Failed to download the input file."}
    
    input_file_path = downloaded_file["file_path"]
    seed = validated_input['seed'] or random.randint(0, 2**32 - 1)
    ddim_steps = validated_input['ddim_steps']
    guidance_scale = validated_input['guidance_scale']
    
    waveform, sr = super_resolution(
        builtModelSr,
        input_file_path,
        seed=seed,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps
    )
    
    # Convert waveform to 16-bit and save to .wav file
    out_wav_path = f"{job['id']}_out.wav"  # Include job ID in the output file name for uniqueness
    out_wav = (waveform[0] * 32767).astype(np.int16).T
    sf.write(out_wav_path, data=out_wav, samplerate=sr)
    
    # Upload the output file and get a URL for download
    output_url = rp_upload.file(out_wav_path, job['id'])
    
    # Optionally, clean up the local output file after uploading
    os.remove(out_wav_path)
    
    return {"output_url": output_url}

# Assuming runpod.serverless.start correctly defined elsewhere
runpod.serverless.start({"handler": handler})
