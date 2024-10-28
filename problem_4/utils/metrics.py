
from dataclasses import dataclass, field
from typing import List


@dataclass
class Metrics:
    loss_: float = 0.0
    rewards_: float = 0.0
    current_episode: int = 0
    current_steps: int = 0
    loss: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    def step(self, loss: float, reward: float):
        self.current_steps += 1
        self.loss_ += loss
        self.rewards_ += reward

    def episode(self):
        self.loss.append(self.loss_)
        self.rewards.append(self.rewards_)
        self.steps.append(self.current_steps)
        self.current_steps = 0
        self.current_episode += 1
        self.rewards_ = 0.0
        self.loss_ = 0.0

    def log(self):
        print(
            f"Episode: {self.current_episode} Loss: {self.loss[-1]}, Reward: {self.rewards[-1]}")
