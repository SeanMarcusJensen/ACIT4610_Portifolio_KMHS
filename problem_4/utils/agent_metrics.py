import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass
class AgentMetrics:
    episode_rewards: list = field(default_factory=list)
    episode_steps: list = field(default_factory=list)

    def plot(self, path: str | None) -> None:
        """ Plot the rewards, steps, and epsilon over episodes with improved aesthetics. """
        episodes = np.arange(1, len(self.episode_rewards) + 1)

        # Ensure all data series are of the same length
        rewards = np.array(self.episode_rewards[:len(episodes)])
        steps = np.array(self.episode_steps[:len(episodes)])

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

        # Titles and legends
        plt.title('Training Metrics')
        fig.tight_layout()  # Adjust layout to not overlap labels
        ax1.legend(loc='upper left')

        if path:
            folder = path.rsplit('/', 1)[0]
            if folder:
                os.makedirs(folder, exist_ok=True)
            plt.savefig(path)
        else:
            plt.show()

