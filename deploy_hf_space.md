---
title: Deploy SQL Debug Env to HF Spaces
description: Step-by-step guide to deploy the environment and then train with GRPO
---

# Deploying the SQL Debug Environment to HF Spaces

## Step 1 — Create the HF Space

Go to https://huggingface.co/new-space and configure:

| Field | Value |
|---|---|
| Space name | `sql-debug-env` |
| SDK | **Docker** |
| Hardware | CPU Basic (free tier is fine for the env) |
| Visibility | Public (required for openenv validate) |

---

## Step 2 — Prepare the Repository

```powershell
# Install the HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Clone the empty Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/sql-debug-env
cd sql-debug-env
```

---

## Step 3 — Copy Environment Files

Copy everything from `sql_env/` into the cloned Space repo:

```powershell
# From your sql_env directory:
Copy-Item -Recurse * "C:\path\to\sql-debug-env\" -Force
```

The Space repo should look like:

```
sql-debug-env/          ← HF Space repo root
├── README.md           ← HF Space card (already has ---yaml--- header)
├── server/
│   └── Dockerfile      ← HF Spaces uses this automatically
├── models.py
├── client.py
├── openenv.yaml
├── server/app.py
├── server/environment.py
├── server/data.py
├── server/graders.py
├── server/rewards.py
└── server/requirements.txt
```

> **Important:** HF Spaces looks for a `Dockerfile` at the repo root OR inside `server/`.
> Our Dockerfile is at `server/Dockerfile`. HF will find it automatically.
> The Dockerfile exposes **port 7860** — this is required by HF Spaces.

---

## Step 4 — Push & Deploy

```powershell
cd sql-debug-env
git add .
git commit -m "Initial SQL Debug OpenEnv environment"


```

HF Spaces will automatically:
1. Detect the Dockerfile
2. Build the Docker image
3. Start the server on port 7860
4. Make it available at `https://YOUR_USERNAME-sql-debug-env.hf.space`

---

## Step 5 — Verify the Deployment

```powershell
$SPACE_URL = "https://YOUR_USERNAME-sql-debug-env.hf.space"

# Health check
Invoke-WebRequest "$SPACE_URL/health" | Select-Object -Expand Content

# List tasks
Invoke-WebRequest "$SPACE_URL/tasks" | Select-Object -Expand Content

# Interactive docs
Start-Process "$SPACE_URL/docs"
```

---

## Step 6 — Run Training Against the HF Space

```powershell
# Point training at the deployed Space
$env:ENV_URL    = "https://YOUR_USERNAME-sql-debug-env.hf.space"
$env:USE_LOCAL_ENV = "false"   # use HTTP client

# Optional: push the trained model automatically
$env:PUSH_TO_HUB = "true"
$env:HF_REPO_ID  = "YOUR_USERNAME/sql-debug-qwen-grpo"

python train_grpo.py --mode train --n-repeats 50
```

Or for faster local training (no network overhead):

```powershell
# Local env (default) — start server first
Start-Job { uvicorn server.app:app --host 0.0.0.0 --port 7860 }
$env:USE_LOCAL_ENV = "true"
python train_grpo.py --mode both --n-repeats 50
```

---

## Hardware Requirements for Training

| GPU | Batch Size | num_generations | use_vllm | ETA (3 epochs) |
|---|---|---|---|---|
| A100 40GB | 1 | 8 | True | ~2h |
| A100 40GB | 1 | 4 | False | ~4h |
| RTX 4090 24GB | 1 | 2 | False | ~6h |
| V100 16GB | 1 | 2 | False | OOM risk — use 4bit |

For 4-bit quantization on smaller GPUs, add to `get_grpo_config()`:
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
# Pass to GRPOTrainer via model_init_kwargs
```

---

## Quick Colab Setup

```python
# In Google Colab (A100 runtime)
!pip install trl transformers torch duckdb pandas pydantic fastapi uvicorn requests
!git clone https://huggingface.co/spaces/YOUR_USERNAME/sql-debug-env sql_env
%cd sql_env

import subprocess, threading
server = threading.Thread(
    target=lambda: subprocess.run(
        ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
    ),
    daemon=True
)
server.start()

import time; time.sleep(3)  # wait for server

# Now run training
!python train_grpo.py --mode both --n-repeats 30
```
