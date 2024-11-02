import gym
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from utils import EpsilonGreedy
import matplotlib.pyplot as plt

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
            plt.savefig(path)
        else:
            plt.show()


class TaxiAgent(ABC):
    def __init__(self) -> None:
        self.metrics = AgentMetrics()
        self._env = gym.make("Taxi-v3")

    def train(self, n_episodes: int, step_limit_per_episode: int) -> AgentMetrics:
        metrics = self.__run(n_episodes, step_limit_per_episode, is_training=True)
        self._save()
        return metrics

    def test(self, n_episodes: int, step_limit_per_episode: int) -> AgentMetrics:
        self._load()
        return self.__run(n_episodes, step_limit_per_episode, is_training=False)

    def __run(self, n_episodes: int, step_limit_per_episode: int, is_training: bool) -> AgentMetrics:
        if not is_training:
            self._env = gym.make("Taxi-v3", render_mode="human")

        for _ in range(n_episodes):
            state = self._env.reset()[0]
            episode_reward = 0.0
            episode_steps = 0
            for _ in range(step_limit_per_episode):
                action = self._get_action(state, is_training)

                next_state, reward, terminated, truncated, _ = self._env.step(action)

                if is_training:
                    self._update(state, action, reward, next_state, terminated or truncated)

                state = next_state
                episode_reward += reward
                episode_steps += 1

                if terminated or truncated:
                    break

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
    def _get_action(self, state: int, is_training: bool) -> int:
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

class BasicQAgent(TaxiAgent):
    def __init__(self, epsilon: EpsilonGreedy, learning_rate: float, discount_factor: float) -> None:
        super().__init__()

        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount = discount_factor
        self._q_table = np.zeros((self._env.observation_space.n, self._env.action_space.n)) # type: ignore

    def _load(self) -> None:
        with open("q_table.npy", "rb") as f:
            self._q_table = np.load(f)

    def _save(self) -> None:
        with open("q_table.npy", "wb") as f:
            np.save(f, self._q_table)

    def _get_action(self, state: int, is_training: bool)-> int:
        if self._epsilon.should_be_random and is_training:
            return self._env.action_space.sample()
        return np.argmax(self._q_table[state]).item()

    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        """
        Formula: Q(a, s) <- Q(a, s) + α[R(s) + γ max_a' Q(a', s') - Q(a, s)]
        """
        if terminated:
            reward = 0
            # Update the epsilon value
            self._epsilon.update()

        self._q_table[state, action] += self._learning_rate * \
                (reward + self._discount * np.max(self._q_table[next_state]) - self._q_table[state, action])

class SarsaAgent(TaxiAgent):
    def __init__(self, epsilon: EpsilonGreedy, learning_rate: float, discount_factor: float) -> None:
        super().__init__()

        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount = discount_factor
        self._q_table = np.zeros((self._env.observation_space.n, self._env.action_space.n)) # type: ignore

    def _load(self) -> None:
        with open("sarsa_q_table.npy", "rb") as f:
            self._q_table = np.load(f)

    def _save(self) -> None:
        with open("sarsa_q_table.npy", "wb") as f:
            np.save(f, self._q_table)

    def _get_action(self, state: int, is_training: bool)-> int:
        if self._epsilon.should_be_random and is_training:
            return self._env.action_space.sample()
        return np.argmax(self._q_table[state]).item()

    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        """
        Formula: Q(a, s) <- (1 - α) Q(a, s) + α[R(s) + γ Q(a', s')]
        """
        if terminated:
            reward = 0
            self._epsilon.update()

        self._q_table[state, action] = (1 - self._learning_rate) * self._q_table[state, action] + \
            self._learning_rate * \
            (reward + self._discount *
             np.max(self._q_table[next_state]))

class DeepQAgent(TaxiAgent):
    def __init__(self, epsilon: EpsilonGreedy, learning_rate: float, discount_factor: float) -> None:
        super().__init__()

        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount = discount_factor
        self._q_table = np.zeros((self._env.observation_space.n, self._env.action_space.n))


if __name__ == "__main__":
    basic = BasicQAgent(EpsilonGreedy(1.0, 0.99, 0.1), 0.5, 0.95)
    basic_metrix= basic.train(1000, 1000)
    basic_metrix.plot(None)

    sarsa = SarsaAgent(EpsilonGreedy(1.0, 0.99, 0.1), 0.5, 0.95)
    metrics = sarsa.train(1000, 1000)
    metrics.plot(None)
    
