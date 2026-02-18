import os
# Ensure we are ONLINE for this script
if "HF_HUB_OFFLINE" in os.environ:
    del os.environ["HF_HUB_OFFLINE"]

from huggingface_hub import snapshot_download

def download_model():
    model_id = "stabilityai/sdxl-turbo"
    print(f"Downloading {model_id}...")

    # Download everything from the repo to a specific local directory
    local_dir = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    snapshot_download(repo_id=model_id)

    print(f"Successfully downloaded to {local_dir}")

if __name__ == "__main__":
    download_model()
