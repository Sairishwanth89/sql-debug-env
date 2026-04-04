from typing import Optional
from pydantic import BaseModel

class MyEnvV4Observation(BaseModel):
    echoed_message: str

class MyEnvV4Result(BaseModel):
    observation: MyEnvV4Observation
    reward: float
    done: bool
    error: Optional[str] = None

class MyEnvV4Action(BaseModel):
    message: str

class MyEnvV4Env:
    """
    Mock Environment matching the sample provided.
    Always acts as a local Python environment, bypassing Docker for fast evaluation testing!
    """

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None):
        return cls()

    async def reset(self) -> MyEnvV4Result:
        return MyEnvV4Result(
            observation=MyEnvV4Observation(echoed_message="[Environment Initialized]"),
            reward=0.0,
            done=False
        )

    async def step(self, action: MyEnvV4Action) -> MyEnvV4Result:
        message = action.message
        
        # Grading Logic provided in standard inference config:
        # "Reward is proportional to message length: reward = len(message) * 0.1"
        reward = len(message) * 0.1
        
        return MyEnvV4Result(
            observation=MyEnvV4Observation(echoed_message=message),
            reward=reward,
            done=False
        )

    async def close(self):
        """Simulate container and socket cleanup"""
        pass
