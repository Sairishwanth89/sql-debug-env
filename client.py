"""
client.py — OpenEnv client for SQL Debug & Data Pipeline Repair.
Provides a typed, sync/async interface that mirrors the EnvClient spec.
"""

from __future__ import annotations
from typing import Optional

from models import SQLDebugAction, SQLDebugObservation, SQLDebugState

try:
    from openenv.core.env_client import EnvClient      # type: ignore
    from openenv.core.client_types import StepResult   # type: ignore

    class SQLDebugEnv(EnvClient[SQLDebugAction, SQLDebugObservation, SQLDebugState]):
        """
        Typed client for the SQL Debug environment.

        Usage (sync):
            with SQLDebugEnv(base_url="http://localhost:7860").sync() as env:
                obs = env.reset(task_id="task1_syntax_fix")
                action = SQLDebugAction(fixed_sql="SELECT ...")
                obs, reward, done, info = env.step(action)

        Usage (async):
            async with SQLDebugEnv(base_url="http://localhost:7860") as env:
                obs = await env.reset()
                result = await env.step(action)
        """

        def _step_payload(self, action: SQLDebugAction) -> dict:
            return action.model_dump()

        def _parse_result(self, payload: dict) -> StepResult:
            obs_data = payload.get("observation", {})
            return StepResult(
                observation=SQLDebugObservation(**obs_data),
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict) -> SQLDebugState:
            return SQLDebugState(**payload)

except ImportError:

    import requests

    class SQLDebugEnv:  # type: ignore[no-redef]
        """
        Lightweight HTTP client (no openenv-core dependency required).

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
            params: dict = {"seed": seed}
            if task_id:
                params["task_id"] = task_id
            r = requests.post(f"{self.base_url}/reset", params=params)
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
