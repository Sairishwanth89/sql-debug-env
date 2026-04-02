"""
inference.py — inference script for SQL Debug & Data Pipeline Repair.

Runs a model (default: gpt-4o-mini) against all 3 tasks using the OpenAI
client API. Reads credentials from environment variables. Produces a
reproducible JSON report with per-task scores.

Usage:
    # Set credentials
    $env:OPENAI_API_KEY = "sk-..."
    # Optional: use a different base URL (e.g. local vLLM)
    $env:OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    python inference.py
    python inference.py --task task1_syntax_fix
    python inference.py --model gpt-4o --output results.json
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

# Make server package importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SQLDebugAction, SQLDebugObservation
from server.environment import SQLDebugEnvironment
from server.data import TASKS


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(obs: SQLDebugObservation) -> str:
    """Convert an observation into a model prompt."""
    schema_lines = []
    for table, cols in obs.schema_info.items():
        col_defs = ", ".join(f"{c['column']} {c['type']}" for c in cols)
        schema_lines.append(f"  {table}({col_defs})")
    schema_str = "\n".join(schema_lines)

    if obs.task_id == "task3_etl_timezone":
        code_section = f"""
## Broken ETL Pipeline Code
```python
{obs.pipeline_code}
```

## Intermediate Outputs (from the BROKEN pipeline)
{json.dumps(obs.intermediate_outputs, indent=2, default=str) if obs.intermediate_outputs else 'Not available'}
"""
        instruction = (
            "Return the COMPLETE corrected Python pipeline code inside a ```python ... ``` block. "
            "Also provide a brief explanation of the root cause (which step is buggy and why) "
            "in a section labelled 'Explanation:'."
        )
    else:
        code_section = f"""
## Broken SQL Query
```sql
{obs.broken_sql}
```
"""
        instruction = (
            "Return ONLY the corrected SQL query inside a ```sql ... ``` block. "
            "Do not include any explanation outside the code block."
        )

    history_section = ""
    if obs.previous_attempts:
        lines = []
        for a in obs.previous_attempts:
            lines.append(f"  Step {a.step}: reward={a.reward:.2f}  SQL: {a.fixed_sql[:120]}...")
        history_section = "\n## Previous Attempts\n" + "\n".join(lines)

    return f"""You are an expert SQL and data engineering debugger.

## Task ({obs.difficulty.upper()})
{obs.task_description}

## Database Schema
{schema_str}
{code_section}{history_section}

## Instructions
{instruction}
"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _extract_sql(text: str, is_pipeline: bool = False) -> str:
    """Extract SQL or Python code from model response."""
    # Try fenced code block first
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
    # Fallback: return the whole response
    return text.strip()


def _extract_explanation(text: str) -> Optional[str]:
    """Extract explanation section from Task 3 response."""
    m = re.search(r"explanation[:\s]+(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Main baseline loop
# ---------------------------------------------------------------------------

def run_baseline(
    model: str = "gpt-4o-mini",
    task_filter: Optional[str] = None,
    output_path: str = "outputs/baseline_results.json",
    max_steps: int = 3,
    seed: int = 42,
) -> dict:
    """
    Run the baseline agent against all (or one) task(s).
    Returns a results dict with per-task scores.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Set it before running baseline.")

    base_url = os.environ.get("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url)

    env = SQLDebugEnvironment()
    results = {
        "model": model,
        "seed": seed,
        "tasks": {},
    }

    target_tasks = [t for t in TASKS if (task_filter is None or t.task_id == task_filter)]

    for task_spec in target_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_spec.task_id} ({task_spec.difficulty})")
        print(f"{'='*60}")

        task_result = {
            "task_id": task_spec.task_id,
            "difficulty": task_spec.difficulty,
            "steps": [],
            "best_reward": 0.0,
            "final_reward": 0.0,
            "done": False,
        }

        obs: SQLDebugObservation = env.reset(seed=seed, task_id=task_spec.task_id)
        done = False
        best_reward = 0.0

        for step_num in range(1, max_steps + 1):
            if done:
                break

            prompt = _build_prompt(obs)
            print(f"\n  Step {step_num}: calling {model}...")

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert SQL debugger. Follow instructions exactly. "
                                "Return only what is asked for — no extra commentary."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
                raw_text = response.choices[0].message.content or ""
            except Exception as e:
                print(f"  API error: {e}")
                raw_text = ""

            is_pipeline = (task_spec.task_id == "task3_etl_timezone")
            fixed_sql = _extract_sql(raw_text, is_pipeline=is_pipeline)
            explanation = _extract_explanation(raw_text) if is_pipeline else None

            action = SQLDebugAction(fixed_sql=fixed_sql, explanation=explanation)
            obs, reward, done, info = env.step(action)

            best_reward = max(best_reward, reward)
            print(f"  Reward: {reward:.4f}  Done: {done}")
            print(f"  Breakdown: {info.get('breakdown', {})}")

            task_result["steps"].append({
                "step": step_num,
                "reward": reward,
                "done": done,
                "breakdown": info.get("breakdown", {}),
                "penalties": info.get("penalties", {}),
                "fixed_sql_preview": fixed_sql[:200],
            })

            time.sleep(0.5)  # rate limiting

        task_result["best_reward"] = round(best_reward, 4)
        task_result["final_reward"] = round(obs.reward or 0.0, 4)
        task_result["done"] = done
        results["tasks"][task_spec.task_id] = task_result

        print(f"\n  >>> Best reward for {task_spec.task_id}: {best_reward:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for tid, tr in results["tasks"].items():
        print(f"  {tid:40s}  best={tr['best_reward']:.4f}  ({tr['difficulty']})")

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline inference for SQL Debug & Data Pipeline Repair OpenEnv"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=["task1_syntax_fix", "task2_join_aggregation", "task3_etl_timezone"],
        help="Run a single task (default: all tasks)",
    )
    parser.add_argument(
        "--output",
        default="outputs/baseline_results.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Max steps per episode (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    run_baseline(
        model=args.model,
        task_filter=args.task,
        output_path=args.output,
        max_steps=args.max_steps,
        seed=args.seed,
    )
