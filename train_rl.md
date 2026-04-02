# RL Training for SQL Debug — GRPO with Qwen2.5-Coder-7B-Instruct

> **Full training script:** [`train_grpo.py`](train_grpo.py)
> **HF Space deployment:** [`deploy_hf_space.md`](deploy_hf_space.md)

---

## Why GRPO, Not DDPG

| | DDPG | GRPO |
|---|---|---|
| Action space | Continuous R^n | Discrete tokens ✅ |
| Value network | Required | Not needed ✅ |
| Gradient signal | Bellman + actor-critic | Group relative ranking ✅ |
| Works for SQL? | ❌ | ✅ |

DDPG is for robot control / trading. SQL token generation is discrete — **always use GRPO or PPO**.

---

## What `train_grpo.py` Contains

| Section | Description |
|---|---|
| 1. Environment | Local DuckDB env or HTTP client pointing at HF Space |
| 2. Model & Tokenizer | `Qwen/Qwen2.5-Coder-7B-Instruct`, left-padding |
| 3. System Prompt | Rules, Response Format (`<think>` + ```sql```), Strategy, Goal |
| 4. Helpers | `build_user_message()`, `extract_sql_from_response()`, `format_messages()` |
| 5. Rollout | `rollout_func()` — plays multi-turn episode, returns padded tensors |
| 6. Reward Fns | `reward_correctness`, `reward_format`, `reward_no_repetition` |
| 7. Dataset | 3 tasks × N repeats → HF `Dataset` with `prompt` + `task_id` columns |
| 8. GRPOConfig | A100-tuned: `num_generations=4`, `bf16=True`, `max_completion_length=1024` |
| 9. Trainer | `GRPOTrainer` with `reward_weights=[0.7, 0.2, 0.1]` |
| 10. Save & Push | `trainer.save_model()` + `push_to_hub()` when `PUSH_TO_HUB=true` |
| 11. Evaluation | Greedy decode, 10 episodes/task, JSON report in `outputs/evals/` |

---

## Quick Start

```powershell
# Install
pip install trl>=0.12.0 transformers>=4.45.0 torch>=2.3.0 duckdb pandas pydantic

# Start local server (terminal 1)
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Train (terminal 2)
python train_grpo.py --mode train --n-repeats 50

# Evaluate trained model
python train_grpo.py --mode eval --output-dir ./sql-debug-qwen-grpo

# Train + eval in one command
python train_grpo.py --mode both
```

---

## System Prompt Structure

```
RULES         — what the agent must/must not do
RESPONSE FORMAT — <think>...</think> then ```sql...```
STRATEGY      — task-specific hints (syntax / JOIN type / timezone)
GOAL          — produce output matching the ground truth exactly
```

The `<think>` block is critical — it teaches chain-of-thought diagnosis before emitting the fix.

---

## Reward Weights

```python
reward_weights = [0.7, 0.2, 0.1]
# 0.7 × reward_correctness  (dense 0.0–1.0 from grader)
# 0.2 × reward_format       (<think> block + ```sql``` present)
# 0.1 × reward_no_repetition (penalty for trivial empty output)
```

---

## Expected Outcomes After Training

| Task | Before (GPT-4o-mini baseline) | After GRPO (estimated) |
|---|---|---|
| task1_syntax_fix | ~0.85 | ~0.95 |
| task2_join_aggregation | ~0.55 | ~0.75 |
| task3_etl_timezone | ~0.25 | ~0.50 |

Use curriculum (train on Task 1+2 first, then add Task 3) for better Hard task improvement.
