
from gym.spaces import Space
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def initialize(self, action_space: Space, observation_space: Space, n_episodes: int, is_training: bool) -> None:
        pass

    @abstractmethod
    def find_action(self, state: int, action_space: Space) -> int:
        pass

    @abstractmethod
    def update(self, state: int, action: int, reward: float, next_state: int, terminal: bool) -> None:
        pass

    @abstractmethod
    def end_of_episode(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass
