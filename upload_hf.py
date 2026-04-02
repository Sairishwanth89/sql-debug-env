import os
from huggingface_hub import HfApi

api = HfApi()

token = os.environ.get("HF_TOKEN")
repo_id = "sai1912/SQL_debug_env_v1"
repo_type = "space"

print("Uploading to HF...")
try:
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        ignore_patterns=["upload_out.txt", "upload_hf.py", ".git", "__pycache__", "outputs","login_output.txt"]
    )
    print("Upload successful!")
except Exception as e:
    print("Error:", e)
