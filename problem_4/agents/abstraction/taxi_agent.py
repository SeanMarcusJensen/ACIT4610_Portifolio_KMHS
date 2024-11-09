import gymnasium as gym
from utils import EpsilonGreedy
from abc import ABC, abstractmethod
from gymnasium.wrappers import RecordVideo
from utils import AgentMetrics
from typing import Callable


class TaxiAgent(ABC):
    """ A base class for Taxi Agents to inherit from.
    Simplifies the process of training and watching the agent play the game.

    Args:
        ABC (_type_): An abstract base class.
    """

    def __init__(self, epsilon: EpsilonGreedy) -> None:
        self._epsilon = epsilon
        self._env = gym.make("Taxi-v3")  # Needed for subclasses to initialize.
        self._actions = {
            "down": 0,
            "up": 1,
            "right": 2,
            "left": 3,
            "pickup": 4,
            "dropoff": 5}
        self._locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B

    def train(self, n_episodes: int, step_limit_per_episode: int, on_episode_do: Callable[[int, int], None] = lambda x, y: None) -> AgentMetrics:
        """ A method to train the agent.

        Args:
            n_episodes (int): Number of episodes to train the agent.
            step_limit_per_episode (int): The maximum number of steps to take in each episode.
            on_episode_do (Callable[[int, int], None], optional): A callable to keep track on episodes. Often used to log the progress. Defaults to lambdax.

        Returns:
            AgentMetrics: The metrics of the agent's performance.
        """
        self._env.close()
        self._env = gym.make("Taxi-v3")
        metrics = self.__run(
            n_episodes, step_limit_per_episode, is_training=True, on_episode_do=on_episode_do)
        self._save()
        self._env.close()
        return metrics

    def watch(self, n_episodes: int, step_limit_per_episode: int) -> AgentMetrics:
        """ A method to watch the agent play the game without training.

        Args:
            n_episodes (int): Number of episodes to watch. 
            step_limit_per_episode (int): The maximum number of steps to take in each episode.

        Returns:
            AgentMetrics: The metrics of the agent's performance.
        """
        self._load()
        self._env.close()
        self._env = gym.make("Taxi-v3", render_mode="human")
        metrics = self.__run(
            n_episodes, step_limit_per_episode, is_training=False)
        self._env.close()
        return metrics

    def record_video(self, n_episodes: int, path: str) -> None:
        """ A method to record a video of the agent playing the game.
        Can be used only after the agent has been trained.

        Args:
            n_episodes (int): Number of episodes to record.
            path (str): The path to save the videos.
        """
        self._load()
        self._env.close()
        self._env = gym.make("Taxi-v3", render_mode="rgb_array")
        self._env = RecordVideo(
            self._env, video_folder=path, disable_logger=True)
        self.__run(n_episodes, 100, is_training=False)
        self._env.close()

    def __run(self, n_episodes: int, step_limit_per_episode: int, is_training: bool, on_episode_do: Callable[[int, int], None] = lambda x, y: None) -> AgentMetrics:
        """ The main loop of running the agent in the environment.

        Args:
            n_episodes (int): _description_
            step_limit_per_episode (int): _description_
            is_training (bool): _description_
            y (None): _description_
            on_episode_do (Callable[[int, int], None], optional): _description_. Defaults to lambdax.

        Returns:
            AgentMetrics: The metrics of the agent's performance.
        """
        import datetime
        metrics = AgentMetrics()
        for episode in range(1, n_episodes + 1):
            start_time = datetime.datetime.now()
            state = self._env.reset()[0]
            episode_reward = 0.0
            episode_steps = 0
            for _ in range(step_limit_per_episode):

                if is_training and self._epsilon.should_be_random:
                    action = self._env.action_space.sample()
                else:
                    action = self._get_action(state)

                next_state, reward, terminated, truncated, _ = self._env.step(
                    action)

                if is_training:
                    self._update(state, action, reward.__float__(),
                                 next_state, terminated or truncated)

                state = next_state
                episode_reward += reward.__float__()
                episode_steps += 1

                if terminated or truncated:
                    break

            end_time = datetime.datetime.now()
            delta = (end_time - start_time).total_seconds()
            metrics.add(episode_reward, episode_steps,
                        self._epsilon.update(), delta)
            on_episode_do(episode, n_episodes)

        return metrics

    @abstractmethod
    def _load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_action(self, state: int) -> int:
        """ Get the action to take in the current state.

        Args:
            state (int): The current state of the environment.

        Raises:
            NotImplementedError: All subclasses must implement this method.

        Returns:
            int: The action to take in the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        """ Update the agent's knowledge based on the environment's response.
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
