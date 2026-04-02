"""
train_grpo.py — Full GRPO training pipeline for SQL Debug & Data Pipeline Repair
using Qwen/Qwen2.5-Coder-7B-Instruct + TRL GRPOTrainer.

Follows the Module 5 pattern from https://github.com/huggingface/openenv-course

Pipeline:
  1. Init environment (local server or HF Space URL)
  2. Init model & tokenizer (Qwen2.5-Coder-7B-Instruct)
  3. Define system prompt (rules, response format, strategy, goal)
  4. Helper functions (prompt builder, SQL extractor)
  5. Rollout function (plays one full episode against the environment)
  6. Reward functions (wraps our grader decomposition into TRL callbacks)
  7. Create dataset (prompts for all 3 tasks × N variants)
  8. Configure GRPO (GRPOConfig)
  9. Create GRPOTrainer and train
  10. Save & push to Hub
  11. Evaluation loop

Usage:
    # Local environment (start server first)
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Training (single GPU A100/H100 recommended)
    python train_grpo.py

    # With HF Space
    ENV_URL=https://your-username-sql-debug-env.hf.space python train_grpo.py

Requirements:
    pip install trl>=0.12.0 transformers>=4.45.0 torch>=2.3.0
    pip install duckdb pandas pydantic requests vllm  # for local env
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Make local env importable ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import SQLDebugEnv
from models import SQLDebugAction, SQLDebugObservation
from server.data import TASKS


# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================

# Point to your deployed HF Space or local server
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# For training we use the local Python environment directly (no HTTP round-trip)
# This is faster and avoids network latency during rollouts.
# Switch to SQLDebugEnv(ENV_URL) if you want to use the HTTP server.
USE_LOCAL_ENV = os.environ.get("USE_LOCAL_ENV", "true").lower() == "true"

if USE_LOCAL_ENV:
    from server.environment import SQLDebugEnvironment
    _SHARED_ENV = SQLDebugEnvironment()  # single instance, reset() per episode
else:
    # HTTP client — point at your HF Space
    _HTTP_CLIENT = SQLDebugEnv(base_url=ENV_URL)


def get_env():
    """Return the environment handle (local or HTTP)."""
    if USE_LOCAL_ENV:
        return _SHARED_ENV
    return _HTTP_CLIENT


# =============================================================================
# 2. MODEL & TOKENIZER
# =============================================================================

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "sai1912/sql-debug-qwen-grpo")

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Required for decoder-only models in GRPO


# =============================================================================
# 3. SYSTEM PROMPT — Rules, Response Format, Strategy, Goal
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert SQL debugger and data engineer. Your goal is to diagnose \
and fix broken SQL queries and ETL pipelines.

RULES:
- Read the broken SQL or pipeline code carefully
- Study the schema — table names, column names, and types matter
- Look for: syntax errors, wrong aliases, wrong JOIN types, type casting bugs
- Your fix must produce exactly the correct output described in the task
- Never use DROP TABLE, DELETE, or TRUNCATE on real data tables
- Do not repeat the same query if it was already rejected

RESPONSE FORMAT:
Always respond with EXACTLY this structure (no deviation):

<think>
[Your step-by-step diagnosis of the bug. Be explicit about what is wrong and why.]
</think>

```sql
[Your complete corrected SQL query here]
```

EXPLANATION (Task 3 only):
[One sentence naming the root cause step and why it causes wrong results]

STRATEGY:
- Task 1 (easy): Look for syntax errors (missing commas) and wrong table aliases
- Task 2 (medium): Check JOIN types — INNER JOIN silently drops NULL-keyed rows
- Task 3 (hard): Trace the timezone handling — CAST(ts AS DATE) strips offset

GOAL:
Return a corrected SQL query (Tasks 1/2) or corrected Python pipeline \
code (Task 3) that produces output matching the ground truth exactly.
"""

# Task-specific addendum injected into user messages
TASK_HINTS = {
    "task1_syntax_fix": (
        "Hint: Check each line of the SELECT clause carefully. "
        "Also verify every table alias used in JOIN conditions matches the FROM clause aliases."
    ),
    "task2_join_aggregation": (
        "Hint: Consider what happens when a JOIN key is NULL. "
        "INNER JOIN silently drops those rows — is that correct for this aggregation?"
    ),
    "task3_etl_timezone": (
        "Hint: The timestamps include timezone offsets like '+05:30'. "
        "What does CAST(ts AS DATE) do to that offset? "
        "Which DuckDB type preserves timezone information?"
    ),
}


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def build_user_message(obs: SQLDebugObservation) -> str:
    """
    Format an observation into a user-turn message.
    Mirrors baseline.py but adds structured context for RL training.
    """
    # Schema block
    schema_lines = []
    for table, cols in obs.schema_info.items():
        col_defs = ", ".join(f"{c['column']} {c['type']}" for c in cols)
        schema_lines.append(f"  {table}({col_defs})")
    schema_str = "\n".join(schema_lines)

    # Code block
    if obs.task_id == "task3_etl_timezone":
        code_block = (
            f"## Broken ETL Pipeline (Python/DuckDB)\n\n"
            f"```python\n{obs.pipeline_code}\n```"
        )
        if obs.intermediate_outputs:
            wrong_output = json.dumps(obs.intermediate_outputs[-1]["rows"][:3], indent=2, default=str)
            code_block += (
                f"\n\n## Step 4 Wrong Output (first 3 rows)\n\n"
                f"```json\n{wrong_output}\n```"
            )
        response_instruction = (
            "Return the COMPLETE corrected Python pipeline code in a "
            "```python ... ``` block. Set EXPLANATION to name the buggy step."
        )
    else:
        code_block = f"## Broken SQL Query\n\n```sql\n{obs.broken_sql}\n```"
        response_instruction = "Return the corrected SQL inside a ```sql ... ``` block."

    # Previous attempts
    history = ""
    if obs.previous_attempts:
        lines = ["\n## Previous Attempts (learn from these)\n"]
        for a in obs.previous_attempts:
            verdict = "CORRECT" if a.reward >= 1.0 else f"reward={a.reward:.2f}"
            preview = a.fixed_sql[:150].replace("\n", " ")
            lines.append(f"  Attempt {a.step} [{verdict}]: {preview}...")
        history = "\n".join(lines)

    hint = TASK_HINTS.get(obs.task_id, "")

    return (
        f"## Task ({obs.difficulty.upper()}): {obs.task_id}\n\n"
        f"{obs.task_description}\n\n"
        f"## Database Schema\n\n{schema_str}\n\n"
        f"{code_block}"
        f"{history}\n\n"
        f"## Instruction\n{response_instruction}\n\n"
        f"{hint}"
    ).strip()


def extract_sql_from_response(text: str, is_pipeline: bool = False) -> str:
    """
    Extract the SQL or Python code block from a model response.
    Falls back to the raw text if no code block found.
    """
    lang = "python" if is_pipeline else "sql"
    patterns = [
        rf"```{lang}\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
        r"```(.*?)```",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return text.strip()


def extract_explanation(text: str) -> Optional[str]:
    """Extract EXPLANATION section (Task 3 root-cause scoring)."""
    m = re.search(r"EXPLANATION[:\s]+(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()[:300]
    # Also check the think block for step identification
    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_m:
        return think_m.group(1).strip()[:300]
    return None


def format_messages(obs: SQLDebugObservation) -> List[Dict[str, str]]:
    """Build the chat message list for the model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_message(obs)},
    ]


# =============================================================================
# 5. ROLLOUT FUNCTION
# =============================================================================

def generate_rollout_completions(trainer: GRPOTrainer, batch_messages: List[List[Dict]]) -> List[Dict]:
    """
    Generate completions using the current policy model via TRL's built-in
    generate_completions utility (vLLM-backed when use_vllm=True).

    Returns a list of dicts with keys: 'text', 'prompt_ids', 'completion_ids', 'logprobs'.
    """
    # Tokenize prompts
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=2048).to(trainer.model.device)

    with torch.no_grad():
        output_ids = trainer.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, (prompt_ids, out_ids) in enumerate(zip(inputs["input_ids"], output_ids)):
        prompt_len = prompt_ids.shape[0]
        completion_ids = out_ids[prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        results.append({
            "text": text,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": None,  # TRL computes logprobs internally
        })
    return results


def rollout_func(
    trainer: GRPOTrainer,
    batch: Dict[str, Any],
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """
    TRL rollout function. Called by GRPOTrainer during training.

    Plays one full episode per row in the batch:
      1. reset() the environment for the task
      2. Generate a fix with the current policy
      3. step() the environment
      4. Repeat up to max_turns (multi-turn RL)

    Returns a batch-format dict that TRL expects.
    """
    env = get_env()
    max_turns = 3  # 3 attempts per training episode (saves compute)

    all_prompt_ids      = []
    all_completion_ids  = []
    all_rewards         = []
    all_task_rewards    = []   # grade component (no penalties)

    task_ids: List[str] = batch["task_id"]

    for task_id in task_ids:
        # ── Episode start ──────────────────────────────────────────────────
        if USE_LOCAL_ENV:
            obs = env.reset(seed=42, task_id=task_id)
        else:
            obs = env.reset(task_id=task_id)

        episode_prompt_ids      = []
        episode_completion_ids  = []
        episode_rewards         = []
        is_pipeline = (task_id == "task3_etl_timezone")
        done = False

        for turn in range(max_turns):
            if done:
                break

            messages = format_messages(obs)
            completions = generate_rollout_completions(trainer, [messages])
            completion = completions[0]

            fixed_sql = extract_sql_from_response(completion["text"], is_pipeline=is_pipeline)
            explanation = extract_explanation(completion["text"])

            action = SQLDebugAction(fixed_sql=fixed_sql, explanation=explanation)

            if USE_LOCAL_ENV:
                obs, reward, done, info = env.step(action)
            else:
                obs, reward, done, info = env.step(action)

            episode_prompt_ids.append(
                tokenizer(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    return_tensors="pt",
                )["input_ids"][0]
            )
            episode_completion_ids.append(completion["completion_ids"])
            episode_rewards.append(reward)

        # Use the best reward in the episode as the final signal
        best_reward = max(episode_rewards) if episode_rewards else 0.0
        all_rewards.extend([best_reward] * len(episode_rewards))
        all_prompt_ids.extend(episode_prompt_ids)
        all_completion_ids.extend(episode_completion_ids)

    # Pad sequences to same length
    max_prompt_len = max(t.shape[0] for t in all_prompt_ids)
    max_comp_len   = max(t.shape[0] for t in all_completion_ids)

    padded_prompts = torch.stack([
        torch.nn.functional.pad(t, (max_prompt_len - t.shape[0], 0), value=tokenizer.pad_token_id)
        for t in all_prompt_ids
    ])
    padded_completions = torch.stack([
        torch.nn.functional.pad(t, (0, max_comp_len - t.shape[0]), value=tokenizer.pad_token_id)
        for t in all_completion_ids
    ])

    return {
        "prompt_ids":     padded_prompts,
        "completion_ids": padded_completions,
        "rewards":        torch.tensor(all_rewards, dtype=torch.float32),
    }


# =============================================================================
# 6. REWARD FUNCTIONS (TRL-style callbacks)
# =============================================================================
# TRL's GRPOTrainer can accept multiple reward_funcs. Each receives
# (completions, prompts, **kwargs) and returns a list of floats.
# We use our grader decomposition to provide multi-signal training.

def _run_grader(completion_text: str, task_id: str, is_pipeline: bool) -> Dict[str, float]:
    """Run the environment grader and return breakdown dict."""
    import duckdb as _duckdb
    from server.data import TASK_MAP
    from server.graders import grade_task1, grade_task2, grade_task3

    task = TASK_MAP[task_id]
    con = _duckdb.connect(":memory:")
    con.execute(task.schema_ddl)
    con.execute(task.seed_sql)
    gt_df = con.execute(task.ground_truth_query).df()

    fixed = extract_sql_from_response(completion_text, is_pipeline=is_pipeline)
    explanation = extract_explanation(completion_text)

    try:
        if task_id == "task1_syntax_fix":
            score, breakdown = grade_task1(fixed, gt_df, con)
        elif task_id == "task2_join_aggregation":
            score, breakdown = grade_task2(fixed, gt_df, con)
        elif task_id == "task3_etl_timezone":
            con.execute(task.schema_ddl)
            con.execute(task.seed_sql)
            score, breakdown = grade_task3(fixed, gt_df, con, explanation)
        else:
            score, breakdown = 0.0, {}
    except Exception:
        score, breakdown = 0.0, {}
    finally:
        con.close()

    return {"score": score, **breakdown}


def reward_correctness(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    Primary reward: overall grader score (0.0–1.0).
    This is the dense, decomposed score from our grader.
    """
    task_ids: List[str] = kwargs.get("task_id", ["task1_syntax_fix"] * len(completions))
    rewards = []
    for text, task_id in zip(completions, task_ids):
        is_pipeline = (task_id == "task3_etl_timezone")
        result = _run_grader(text, task_id, is_pipeline)
        rewards.append(result["score"])
    return rewards


def reward_parses(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    Shaping reward: did the SQL parse? (+0.1 bonus).
    Encourages the model to produce syntactically valid SQL even when
    semantics are wrong — important early in training.
    """
    task_ids: List[str] = kwargs.get("task_id", ["task1_syntax_fix"] * len(completions))
    rewards = []
    for text, task_id in zip(completions, task_ids):
        is_pipeline = (task_id == "task3_etl_timezone")
        result = _run_grader(text, task_id, is_pipeline)
        rewards.append(result.get("parses", 0.0))
    return rewards


def reward_format(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    Format reward: did the model use a ```sql ... ``` block?
    This teaches the model the required response format.
    """
    rewards = []
    task_ids: List[str] = kwargs.get("task_id", ["task1_syntax_fix"] * len(completions))
    for text, task_id in zip(completions, task_ids):
        lang = "python" if task_id == "task3_etl_timezone" else "sql"
        has_block = bool(re.search(rf"```{lang}", text, re.IGNORECASE))
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        score = (0.5 if has_block else 0.0) + (0.5 if has_think else 0.0)
        rewards.append(score)
    return rewards


def reward_no_repetition(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    Penalise repetitive/trivial outputs (empty or < 10 chars of code).
    """
    rewards = []
    task_ids: List[str] = kwargs.get("task_id", ["task1_syntax_fix"] * len(completions))
    for text, task_id in zip(completions, task_ids):
        is_pipeline = (task_id == "task3_etl_timezone")
        code = extract_sql_from_response(text, is_pipeline=is_pipeline)
        penalty = -0.3 if len(code) < 10 else 0.0
        rewards.append(penalty)
    return rewards


# =============================================================================
# 7. CREATE DATASET
# =============================================================================

def create_training_dataset(n_repeats: int = 50) -> Dataset:
    """
    Build a training dataset from the 3 tasks.
    Each task is repeated n_repeats times so the model sees diverse episodes.
    The 'prompt' column is a pre-tokenised chat string; 'task_id' is metadata
    passed through to reward functions via kwargs.
    """
    env = get_env()
    rows = []

    for task in TASKS:
        obs = env.reset(seed=42, task_id=task.task_id) if USE_LOCAL_ENV else env.reset(task_id=task.task_id)
        messages = format_messages(obs)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        for i in range(n_repeats):
            rows.append({
                "prompt":   prompt_text,
                "task_id":  task.task_id,
                "difficulty": task.difficulty,
                # Seed varies so GRPO sees slightly different phrasings across epochs
                "seed": 42 + i,
            })

    dataset = Dataset.from_list(rows)
    print(f"Dataset created: {len(dataset)} rows "
          f"({n_repeats} × {len(TASKS)} tasks)")
    return dataset


# =============================================================================
# 8. CONFIGURE GRPO TRAINING
# =============================================================================

def get_grpo_config(output_dir: str = "./sql-debug-qwen-grpo") -> GRPOConfig:
    """
    Return a GRPOConfig tuned for Qwen2.5-Coder-7B on a single A100/H100 40GB.
    Reduce per_device_train_batch_size and num_generations for smaller GPUs.
    """
    return GRPOConfig(
        # ── Output ──────────────────────────────────────────────────────────
        output_dir=output_dir,
        run_name="sql-debug-grpo-qwen25coder7b",

        # ── Training schedule ───────────────────────────────────────────────
        num_train_epochs=3,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # ── Batch & memory ──────────────────────────────────────────────────
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        gradient_checkpointing=True,
        bf16=True,

        # ── GRPO-specific ────────────────────────────────────────────────────
        num_generations=4,            # G: candidates per prompt to compare
        max_prompt_length=2048,
        max_completion_length=1024,   # SQL fixes can be verbose

        # ── vLLM for fast generation (requires vllm package) ─────────────────
        # Set use_vllm=False if not using vLLM (much slower but works on any GPU)
        use_vllm=False,               # set True on A100+ with vllm installed
        # vllm_mode="colocate",
        # vllm_gpu_memory_utilization=0.2,

        # ── Logging ──────────────────────────────────────────────────────────
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        report_to="none",             # set "wandb" or "tensorboard" as needed

        # ── Hub ───────────────────────────────────────────────────────────────
        push_to_hub=False,            # set True to auto-push checkpoints
        hub_model_id=HF_REPO_ID,
    )


# =============================================================================
# 9. CREATE TRAINER & TRAIN
# =============================================================================

def build_trainer(
    dataset: Dataset,
    grpo_config: GRPOConfig,
) -> GRPOTrainer:
    """
    Instantiate GRPOTrainer with:
      - The base model (Qwen2.5-Coder-7B-Instruct)
      - 3 reward functions (correctness, format, no-repetition)
      - The rollout function that drives environment interaction
      - The training dataset
    """
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        # Multiple reward functions — TRL sums them with equal weight by default.
        # You can pass reward_weights=[0.7, 0.2, 0.1] to control contribution.
        reward_funcs=[
            reward_correctness,   # primary: correctness score 0.0–1.0
            reward_format,        # shaping: forces <think> + ```sql``` format
            reward_no_repetition, # penalty: discourages trivial empty outputs
        ],
        reward_weights=[0.7, 0.2, 0.1],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        # rollout_func: commented out here because TRL ≥0.12 uses reward_funcs
        # directly for non-interactive tasks. Use rollout_func for multi-turn.
        # rollout_func=rollout_func,  # uncomment for multi-turn RL
    )
    return trainer


def train(n_repeats: int = 50):
    """Main training entry point."""
    print("=" * 60)
    print(f"Model:    {MODEL_NAME}")
    print(f"Env URL:  {ENV_URL if not USE_LOCAL_ENV else 'local'}")
    print(f"Tasks:    {[t.task_id for t in TASKS]}")
    print("=" * 60)

    dataset = create_training_dataset(n_repeats=n_repeats)
    grpo_config = get_grpo_config()
    trainer = build_trainer(dataset, grpo_config)

    print("\nStarting GRPO training…")
    trainer.train()

    return trainer


# =============================================================================
# 10. SAVE & PUSH TO HUB
# =============================================================================

def save_and_push(trainer: GRPOTrainer, output_dir: str = "./sql-debug-qwen-grpo"):
    """Save the trained model locally and optionally push to the Hub."""
    print(f"\nSaving model to {output_dir}…")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    push = os.environ.get("PUSH_TO_HUB", "false").lower() == "true"
    if push:
        print(f"Pushing to Hub: {HF_REPO_ID}")
        trainer.push_to_hub(
            repo_id=HF_REPO_ID,
            commit_message="GRPO-trained SQL debug model",
        )
        print(f"Model available at: https://huggingface.co/{HF_REPO_ID}")
    else:
        print(f"Set PUSH_TO_HUB=true to push to {HF_REPO_ID}")


# =============================================================================
# 11. EVALUATION
# =============================================================================

@dataclass
class EvalResult:
    task_id: str
    difficulty: str
    n_episodes: int
    mean_reward: float
    best_reward: float
    n_solved: int  # episodes with reward >= 1.0


def evaluate(
    model_path: str = "./sql-debug-qwen-grpo",
    n_episodes: int = 10,
    max_steps: int = 5,
) -> List[EvalResult]:
    """
    Evaluate the trained model against all 3 tasks.
    Loads the fine-tuned model and runs n_episodes per task.
    """
    print(f"\n{'='*60}\nEVALUATION — {model_path}\n{'='*60}")

    eval_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    eval_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eval_model.eval()

    env = get_env()
    results: List[EvalResult] = []

    for task in TASKS:
        episode_rewards = []
        n_solved = 0

        for ep in range(n_episodes):
            seed = 1000 + ep  # different seeds from training
            obs = env.reset(seed=seed, task_id=task.task_id) if USE_LOCAL_ENV \
                else env.reset(task_id=task.task_id)

            best_reward = 0.0
            done = False
            is_pipeline = (task.task_id == "task3_etl_timezone")

            for step in range(max_steps):
                if done:
                    break

                messages = format_messages(obs)
                prompt_text = eval_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = eval_tokenizer(
                    prompt_text, return_tensors="pt", truncation=True, max_length=2048
                ).to(eval_model.device)

                with torch.no_grad():
                    output_ids = eval_model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.0,  # greedy for eval
                        do_sample=False,
                        pad_token_id=eval_tokenizer.eos_token_id,
                    )

                prompt_len = inputs["input_ids"].shape[1]
                completion = eval_tokenizer.decode(
                    output_ids[0][prompt_len:], skip_special_tokens=True
                )

                fixed_sql = extract_sql_from_response(completion, is_pipeline=is_pipeline)
                explanation = extract_explanation(completion)
                action = SQLDebugAction(fixed_sql=fixed_sql, explanation=explanation)

                obs, reward, done, info = env.step(action) if USE_LOCAL_ENV \
                    else env.step(action)

                best_reward = max(best_reward, reward)

            episode_rewards.append(best_reward)
            if best_reward >= 1.0:
                n_solved += 1

        mean_r = sum(episode_rewards) / len(episode_rewards)
        best_r = max(episode_rewards)

        result = EvalResult(
            task_id=task.task_id,
            difficulty=task.difficulty,
            n_episodes=n_episodes,
            mean_reward=round(mean_r, 4),
            best_reward=round(best_r, 4),
            n_solved=n_solved,
        )
        results.append(result)
        print(f"  {task.task_id:40s}  mean={mean_r:.4f}  best={best_r:.4f}  "
              f"solved={n_solved}/{n_episodes}")

    # Write evaluation report
    report = {
        "model": model_path,
        "n_episodes": n_episodes,
        "tasks": [r.__dict__ for r in results],
    }
    os.makedirs("outputs/evals", exist_ok=True)
    report_path = f"outputs/evals/eval_{int(time.time())}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nEval report saved: {report_path}")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO training for SQL Debug environment")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="train")
    parser.add_argument("--n-repeats", type=int, default=50, help="Dataset repeats per task")
    parser.add_argument("--n-episodes", type=int, default=10, help="Eval episodes per task")
    parser.add_argument("--output-dir", default="./sql-debug-qwen-grpo")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        trainer = train(n_repeats=args.n_repeats)
        save_and_push(trainer, output_dir=args.output_dir)

    if args.mode in ("eval", "both"):
        evaluate(model_path=args.output_dir, n_episodes=args.n_episodes)
