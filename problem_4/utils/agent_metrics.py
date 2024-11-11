import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class AgentMetrics:
    episode_rewards: list = field(default_factory=list)
    episode_steps: list = field(default_factory=list)
    episode_epsilon: list = field(default_factory=list)
    episode_time: list = field(default_factory=list)

    def add(self, rewards: float, steps: int, epsilon: float, elapsed: float) -> None:
        self.episode_time.append(elapsed)
        self.episode_rewards.append(rewards)
        self.episode_steps.append(steps)
        self.episode_epsilon.append(epsilon)

    def as_dict(self) -> dict:
        return {
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'epsilon': self.episode_epsilon,
            'time': self.episode_time,
        }

    def as_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.as_dict())

    def save(self, path: str) -> None:
        self.as_pandas().to_csv(path, index=False)
