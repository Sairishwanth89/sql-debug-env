"""
models.py — SQL Debug & Data Pipeline Repair OpenEnv
Typed Pydantic models for Observation, Action, and State.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base stubs (mirrors openenv-core base classes so this module is importable
# without openenv-core installed, while still being fully compatible when it
# is installed).
# ---------------------------------------------------------------------------

try:
    from openenv.core.env_server import Action, Observation, State  # type: ignore
except ImportError:
    class _Base(BaseModel):
        pass
    Action = _Base      # type: ignore[misc,assignment]
    Observation = _Base # type: ignore[misc,assignment]
    State = _Base       # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class PreviousAttempt(BaseModel):
    """Log of a single previous attempt by the agent."""
    step: int
    fixed_sql: str
    reward: float
    info: Dict[str, Any] = Field(default_factory=dict)


class SQLDebugObservation(Observation):
    """
    What the agent sees at each step.

    For Tasks 1 & 2 the key field is `broken_sql`.
    For Task 3 the key field is `pipeline_code`; `intermediate_outputs`
    contains the (wrong) intermediate DataFrames serialised as list-of-dicts.
    """

    # ── Episode metadata ────────────────────────────────────────────────────
    task_id: str = Field(description="Which task this episode runs (task1/task2/task3)")
    task_description: str = Field(description="Natural-language goal the agent must achieve")
    difficulty: str = Field(description="easy | medium | hard")

    # ── Problem payload ─────────────────────────────────────────────────────
    broken_sql: Optional[str] = Field(
        default=None,
        description="Broken SQL string — present for Tasks 1 & 2",
    )
    pipeline_code: Optional[str] = Field(
        default=None,
        description="4-step ETL pipeline Python string — present for Task 3",
    )
    intermediate_outputs: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Wrong intermediate outputs from each pipeline step (Task 3)",
    )

    # ── Schema context ───────────────────────────────────────────────────────
    schema_info: Dict[str, List[Dict[str, str]]] = Field(
        description="Table name → list of {column, type} dicts"
    )

    # ── Progress ─────────────────────────────────────────────────────────────
    step_number: int = Field(default=0, description="Current attempt number (0-indexed)")
    max_steps: int = Field(default=5, description="Maximum attempts allowed")
    previous_attempts: List[PreviousAttempt] = Field(default_factory=list)

    # ── OpenEnv required fields ──────────────────────────────────────────────
    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SQLDebugAction(Action):
    """
    What the agent submits each step.

    `fixed_sql` is required for all tasks.
    For Task 3, `fixed_sql` should contain the COMPLETE corrected pipeline
    Python code (not just a patch).
    `explanation` is optional but scored separately for Task 3's root-cause
    component (+0.15 if it correctly names Step 2 as the bug location).
    """

    fixed_sql: str = Field(
        description=(
            "Corrected SQL string (Tasks 1 & 2) or corrected full "
            "pipeline Python code string (Task 3)"
        )
    )
    explanation: Optional[str] = Field(
        default=None,
        description=(
            "Optional natural-language explanation of the root cause. "
            "Scored for Task 3 root-cause identification (+0.15)."
        ),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SQLDebugState(State):
    """
    Full internal state — used by state() and by the baseline script for
    logging; also inspected by openenv validate.
    """

    task_id: str = Field(default="")
    seed: int = Field(default=42)
    step_count: int = Field(default=0)
    max_steps: int = Field(default=5)
    episode_id: Optional[str] = Field(default=None)
    current_score: float = Field(default=0.0, description="Best score seen so far this episode")
    reward_history: List[float] = Field(default_factory=list)
    done: bool = Field(default=False)
