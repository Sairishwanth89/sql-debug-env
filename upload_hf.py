import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

api = HfApi()

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
repo_id = "sai1912/SQL_debug_env_v1"
repo_type = "space"

print(f"Uploading to HF Repository: {repo_id}...")
try:
    # Ensure the repo exists as a Space
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
    except Exception:
        print(f"Space {repo_id} not found. Creating it...")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, token=token, space_sdk="docker")

    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        ignore_patterns=["upload_out.txt", "upload_hf.py", ".git", "__pycache__", "outputs", "login_output.txt", ".env"]
    )
    print("Upload successful!")
except Exception as e:
    print("Error:", e)
