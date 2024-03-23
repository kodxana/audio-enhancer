from huggingface_hub import hf_hub_download

def download_checkpoint(checkpoint_name="basic"):
    model_id = "haoheliu/wellsolve_audio_super_resolution_48k"

    # Download the checkpoint file. This will use the default cache directory.
    checkpoint_path = hf_hub_download(
        repo_id=model_id, filename=f"{checkpoint_name}.pth"
    )
    return checkpoint_path

if __name__ == "__main__":
    checkpoint_path = download_checkpoint()
    print(f"Checkpoint downloaded and cached at: {checkpoint_path}")
