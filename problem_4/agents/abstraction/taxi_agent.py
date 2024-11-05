import gymnasium as gym
from utils import EpsilonGreedy
from abc import ABC, abstractmethod
from gymnasium.wrappers import RecordVideo 
from utils import AgentMetrics


class TaxiAgent(ABC):
    def __init__(self, epsilon: EpsilonGreedy) -> None:
        self.metrics = AgentMetrics()
        self._epsilon = epsilon

        self._env = gym.make("Taxi-v3") # Needed for subclasses to initialize.

        self._actions = {"down": 0, "up": 1, "right": 2, "left": 3, "pickup": 4, "dropoff": 5}
        self._locations = [(0, 0), (0, 4), (4, 0), (4, 3)] # R, G, Y, B

    def train(self, n_episodes: int, step_limit_per_episode: int) -> AgentMetrics:
        metrics = self.__run(n_episodes, step_limit_per_episode, is_training=True)
        self._save()
        return metrics

    def watch(self, n_episodes: int, step_limit_per_episode: int) -> AgentMetrics:
        self._load()
        self._env = gym.make("Taxi-v3", render_mode="human")
        return self.__run(n_episodes, step_limit_per_episode, is_training=False)

    def record_video(self, n_episodes: int, path: str) -> None:
        self._load()
        self._env = gym.make("Taxi-v3", render_mode="rgb_array")
        self._env = RecordVideo(self._env, video_folder=path, disable_logger=True)
        self.__run(n_episodes, 100, is_training=False) 

    def __run(self, n_episodes: int, step_limit_per_episode: int, is_training: bool) -> AgentMetrics:
        for _ in range(n_episodes):
            state = self._env.reset()[0]
            episode_reward = 0.0
            episode_steps = 0
            for _ in range(step_limit_per_episode):

                if is_training and self._epsilon.should_be_random:
                    action = self._get_action(state)
                else:
                    action = self._get_action(state)

                next_state, reward, terminated, truncated, _ = self._env.step(action)

                if is_training:
                    self._update(state, action, reward.__float__(), next_state, terminated or truncated)

                state = next_state
                episode_reward += reward.__float__()
                episode_steps += 1

                if terminated or truncated:
                    break

            self._epsilon.update()
            self.metrics.episode_rewards.append(episode_reward)
            self.metrics.episode_steps.append(episode_steps)

        return self.metrics

    @abstractmethod
    def _load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_action(self, state: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        """
        Args:
            state (int): The current state of the environment.
            action (int): The action taken in the current state.
            reward (float): The reward received from the environment.
            next_state (int): The next state of the environment.
            terminated (bool): Whether the episode has terminated - Termination is true on final state!
        """
        raise NotImplementedError

    def __del__(self) -> None:
        try:
            self._env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")
