import pickle
import numpy as np
from gym.spaces import Space
from .abstraction import Agent
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from utils import Metrics, EpsilonGreedy


class BasicQLearningAgent(Agent):
    """ Q-Learning Algorithm

    Q-Learning is a model-free reinforcement learning policy
    which will find the next best action to take in a given a state.
    It chooses this action at random and aims to maximize the reward.
    """

    def __init__(self, EPSILON: float = 1.0, EPSILON_DECAY: float = 0.996, LEARNING_RATE: float = 0.1, DISCOUNT_FACTOR: float = 0.95, EPSILON_MIN: float = 0.01):
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR

        self.__epsilon = EpsilonGreedy(
            EPSILON, EPSILON_DECAY, EPSILON_MIN)

        self.__metrics = Metrics()

        self.__q_table: NDArray[np.float64] | None = None
        self.__rewards: NDArray[np.float64] | None = None
        self.__current_reward = 0
        self.__current_episode = 0
        self.__is_training = True

    def load(self) -> None:
        with open('q_table.pkl', 'rb') as f:
            self.__q_table = pickle.load(f)
            f.close()

    def save(self) -> None:
        assert self.__q_table is not None, "Q-Table is not initialized."
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.__q_table, f)
            f.close()

    def initialize(self, action_space: Space, observation_space: Space, n_episodes: int, is_training: bool = True) -> None:
        if not is_training:
            self.load()
        else:
            self.__q_table = np.zeros((observation_space.n, action_space.n))

        self.__rewards = np.zeros(n_episodes)
        self.__is_training = is_training

    def find_action(self, state: int, action_space: Space) -> int:
        assert self.__q_table is not None, "Q-Table is not initialized."
        action: int = 0

        if self.__is_training and self.__epsilon.should_be_random:
            action = action_space.sample()
        else:
            action = np.argmax(self.__q_table[state, :]).item()

        return action

    def update(self, state: int, action: int, reward: float, next_state: int, terminal: bool) -> None:
        if not self.__is_training:
            return

        assert self.__q_table is not None, "Q-Table is not initialized."
        self.__q_table[state, action] = reward + \
            np.max(self.__q_table[next_state])
        self.__current_reward += reward

    def end_of_episode(self) -> None:
        assert self.__rewards is not None, "Rewards array is not initialized."

        self.__epsilon.update(0)

        if not self.__is_training:
            return

        self.__rewards[self.__current_episode] = self.__current_reward
        self.__metrics.step(0, self.__current_reward)
        self.__current_reward = 0
        self.__current_episode += 1

    def plot(self, save_location: str | None = None) -> None:
        """ Plot the rewards, steps, and epsilon over episodes with improved aesthetics. """
        episodes = np.arange(1, self.__metrics.current_episode + 1)

        # Ensure all data series are of the same length
        rewards = np.array(self.__metrics.rewards[:len(episodes)])
        steps = np.array(self.__metrics.steps[:len(episodes)])
        epsilon_history = np.array(self.__epsilon.history[:len(episodes)])

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

        # Create a second y-axis for Epsilon values
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Epsilon (log scale)', color=color)
        epsilon_log = np.array(epsilon_history)
        ax2.plot(episodes, epsilon_log, label='Epsilon (log)',
                 color=color, linestyle='--', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)

        # Titles and legends
        plt.title('Training Metrics: Rewards, Epsilon, and Steps over Episodes')
        fig.tight_layout()  # Adjust layout to not overlap labels
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        if save_location:
            plt.savefig(save_location + '/training_metrics/q-learning.png')

        plt.show()
