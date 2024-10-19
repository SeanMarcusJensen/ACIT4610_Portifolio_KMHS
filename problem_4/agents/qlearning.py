from gym.spaces import Space
import pickle
import numpy as np
from numpy.typing import NDArray
from .abstraction import Agent


class BasicQLearningAgent(Agent):
    """ Q-Learning Algorithm

    Q-Learning is a model-free reinforcement learning policy
    which will find the next best action to take in a given a state.
    It chooses this action at random and aims to maximize the reward.
    """

    def __init__(self, EPSILON: float = 1.0, EPSILON_DECAY: float = 0.001, LEARNING_RATE: float = 0.1, DISCOUNT_FACTOR: float = 0.9, EPSILON_MIN: float = 0.01):
        self.EPSILON = EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.EPSILON_MIN = EPSILON_MIN

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

        if self.__is_training and (np.random.uniform(0, 1) < self.EPSILON):
            action = action_space.sample()
        else:
            action = np.argmax(self.__q_table[state, :]).item()

        return action

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        if not self.__is_training:
            return

        assert self.__q_table is not None, "Q-Table is not initialized."
        self.__q_table[state, action] = reward + \
            np.max(self.__q_table[next_state])
        self.__current_reward += reward

    def end_of_episode(self) -> None:
        assert self.__rewards is not None, "Rewards array is not initialized."

        self.EPSILON = max(self.EPSILON_MIN, self.EPSILON_DECAY * self.EPSILON)

        if not self.__is_training:
            return

        self.__rewards[self.__current_episode] = self.__current_reward
        self.__current_reward = 0
        self.__current_episode += 1
