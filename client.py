"""
client.py — OpenEnv client for SQL Debug & Data Pipeline Repair.
Provides a typed, sync/async interface that mirrors the EnvClient spec.
"""

from __future__ import annotations
from typing import Optional

from models import SQLDebugAction, SQLDebugObservation, SQLDebugState

import requests

class SQLDebugEnv:
    """
    Lightweight HTTP client.
    
    Usage:
        env = SQLDebugEnv(base_url="http://localhost:7860")
        obs_data = env.reset(task_id="task1_syntax_fix")
        result = env.step(SQLDebugAction(fixed_sql="SELECT ..."))
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(
        self,
        seed: int = 42,
        task_id: Optional[str] = None,
    ) -> SQLDebugObservation:
        payload: dict = {"seed": seed}
        if task_id:
            payload["task_id"] = task_id
        r = requests.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return SQLDebugObservation(**r.json())

    def step(
        self,
        action: SQLDebugAction,
    ) -> tuple[SQLDebugObservation, float, bool, dict]:
        r = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
        )
        r.raise_for_status()
        d = r.json()
        obs = SQLDebugObservation(**d["observation"])
        return obs, d["reward"], d["done"], d.get("info", {})

    def state(self) -> SQLDebugState:
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return SQLDebugState(**r.json())

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
