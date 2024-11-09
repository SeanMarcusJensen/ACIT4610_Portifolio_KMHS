import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import pandas as pd
import datetime


@dataclass
class AgentMetrics:
    episode_rewards: list = field(default_factory=list)
    episode_steps: list = field(default_factory=list)
    episode_epsilon: list = field(default_factory=list)
    episode_time: list = field(default_factory=list)
    start_time = datetime.datetime.now()

    def add(self, rewards: float, steps: int, epsilon: float) -> None:
        self.episode_time.append(datetime.datetime.now())
        self.episode_rewards.append(rewards)
        self.episode_steps.append(steps)
        self.episode_epsilon.append(epsilon)

    def as_dict(self) -> dict:
        elapsed_time = np.diff([self.start_time, *self.episode_time]) # type: ignore
        elapsed_time = map(lambda x: x.total_seconds(), elapsed_time)
        return {
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'epsilon': self.episode_epsilon,
            'time': elapsed_time,
        }

    def as_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.as_dict())
    
    def save(self, path: str) -> None:
        self.as_pandas().to_csv(path, index=False)

    def plot(self, path: str | None = None, name: str | None = None) -> None:
        """ Plot the rewards, steps, and epsilon over episodes with improved aesthetics. """
        episodes = np.arange(1, len(self.episode_rewards))

        # Ensure all data series are of the same length
        rewards = np.array(self.episode_rewards[:len(episodes)])
        steps = np.array(self.episode_steps[:len(episodes)])
        epsilon = np.array(self.episode_epsilon[:len(episodes)])

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot rewards per episode
        color = 'tab:blue'
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Rewards', color=color)
        ax1.plot(episodes, rewards, label='Rewards per Episode',
                 color=color, linewidth=2)

        # Compute a moving average for rewards (window size of 50).
        window_size = 50
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(
                window_size)/window_size, mode='valid')
            ax1.plot(episodes[:len(moving_avg)], moving_avg,
                     label=f'Moving Avg (Window: {window_size})', color='orange', linewidth=2)

        # Scale steps to match the range of rewards
        ax1.plot(episodes, steps, label='Scaled Steps',
                 color='green', linewidth=2)

        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon', color='tab:red')
        ax2.plot(episodes, epsilon, '--', label='Epsilon',
                 color='tab:red', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_xlim(xmin=0, xmax=len(episodes))
        ax2.set_ylim(ymin=0, ymax=1.0)

        plt.title('Training Metrics ' + (name or ''))
        fig.tight_layout()  # Adjust layout to not overlap labels
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        if path:
            folder = path.rsplit('/', 1)[0]
            if folder:
                os.makedirs(folder, exist_ok=True)
            plt.savefig(path)
        else:
            plt.show()
