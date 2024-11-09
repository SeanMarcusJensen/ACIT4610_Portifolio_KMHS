import os
from .agent_metrics import AgentMetrics
import pandas as pd
import matplotlib.pyplot as plt

class AgentMetricsComparer:
    def __init__(self, metrics: dict[str, AgentMetrics] | None = None, dir_name: str | None = None):
        self.data: dict[str, pd.DataFrame] = {}
        if metrics is not None:
            self.data = {k:v.as_pandas() for k, v in metrics.items()}
        elif dir_name is not None:
            for file in os.listdir(dir_name):
                file_name = os.fsdecode(file)
                if file_name.endswith('.csv'):
                    key = file_name.removesuffix('.csv')
                    value = pd.read_csv(os.path.join(dir_name, file))
                    self.data[key] = value
        else:
            raise ValueError("Need to specify metrics or dir_name.")

        assert len(self.data) > 2, "Need two or more metrics to compare."

    def plot(self, key):
        # Ensure all data series are of the same length
        import numpy as np
        data = self.data[key]
        episodes = len(data)

        rewards = np.array(data.loc[:episodes, 'rewards'])
        steps = np.array(data.loc[:episodes, 'steps'])
        epsilon = np.array(data.loc[:episodes, 'epsilon'])

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot rewards per episode
        color = 'tab:blue'
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Rewards', color=color)
        ax1.plot(rewards, label='Rewards per Episode',
                 color=color, linewidth=2)

        # Compute a moving average for rewards (window size of 50).
        window_size = 50
        if len(rewards) >= window_size:
            ax1.plot(data['rewards'].rolling(window_size).mean(),
                     label=f'Moving Avg (Window: {window_size})', color='orange', linewidth=2)

        # Scale steps to match the range of rewards
        ax1.plot(steps, label='Scaled Steps',
                 color='green', linewidth=2)

        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon', color='tab:red')
        ax2.plot(epsilon, '--', label='Epsilon',
                 color='tab:red', linewidth=2)

        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_xlim(xmin=0, xmax=episodes)
        ax2.set_ylim(ymin=0, ymax=1.0)

        plt.title('Training Metrics ' + (key or ''))
        fig.tight_layout()  # Adjust layout to not overlap labels
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

    def compare_time(self, keys: list[str], window: int = 50):
        fig, axs = plt.subplots(1, len(keys), figsize=(15, 6))

        for index, key in enumerate(keys):
            frame = self[key]
            frame_len = len(frame)
            random = self['random']
            heuristics = self['heuristic']
            axs[index].set_title(f'Time Comparison (Model: {key.title()})')
            axs[index].plot(random.loc[1:frame_len, 'time'].rolling(window).mean(), label='random')
            axs[index].plot(heuristics.loc[1:frame_len, 'time'].rolling(window).mean(), label='heuristic')
            axs[index].plot(frame.loc[1:frame_len, 'time'].rolling(window).mean(), label=key)
            axs[index].legend()
            axs[index].set_xlabel('Episodes')
            axs[index].set_ylabel(f'Avg Time (W: {window})')
            axs[index].grid(True)

        fig.tight_layout()
        plt.show()
    
    def compare_rewards(self, keys: list[str]):
        """ Plots the cummulative rewards for the agent.
        """
        fig, axs = plt.subplots(1, len(keys), figsize=(15, 6))

        for index, key in enumerate(keys):
            frame = self[key]
            frame_len = len(frame)
            random = self['random']
            heuristics = self['heuristic']
            axs[index].set_title(f'Rewards Comparison (Model: {key.title()})')
            axs[index].plot(random.loc[:frame_len, 'rewards'].cumsum(), label='random')
            axs[index].plot(heuristics.loc[:frame_len, 'rewards'].cumsum(), label='heuristic')
            axs[index].plot(frame.loc[:frame_len, 'rewards'].cumsum(), label=key)
            axs[index].legend()
            axs[index].set_xlabel('Episodes')
            axs[index].set_ylabel('Cummulatice Rewards')
            axs[index].grid(True)

        fig.tight_layout()
        plt.show()
    
    def compare_steps(self, keys: list[str], window: int = 50):
        fig, axs = plt.subplots(1, len(keys), figsize=(15, 6), sharey=True)

        for index, key in enumerate(keys):
            frame = self[key]
            frame_len = len(frame)
            random = self['random']
            heuristics = self['heuristic']
            axs[index].set_title(f'Steps Comparison (Model: {key.title()})')
            axs[index].plot(random.loc[:frame_len, 'steps'].rolling(window).mean(), label='random')
            axs[index].plot(heuristics.loc[:frame_len, 'steps'].rolling(window).mean(), label='heuristic')
            axs[index].plot(frame.loc[:frame_len, 'steps'].rolling(window).mean(), label=key)
            axs[index].legend()
            axs[index].set_xlabel('Episodes')
            axs[index].set_ylabel(f'Avg Steps (w: {window})')
            axs[index].grid(True)

        fig.tight_layout()
        plt.show()
    
    def _get_figure(self, fig_size=(10, 6)):
        fig, ax = plt.subplots(figsize=fig_size)
        return fig, ax
    
    def __getitem__(self, key):
        return self.data[key]

if __name__ == "__main__":
    plotter = AgentMetricsComparer(dir_name='static/metrics')
    plotter.compare_steps(['basic', 'sarsa', 'dql'])
    plotter.compare_rewards(['basic', 'sarsa', 'dql'])
    plotter.compare_time(['basic', 'sarsa', 'dql'])

