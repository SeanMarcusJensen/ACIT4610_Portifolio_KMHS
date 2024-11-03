import os
import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo 
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from utils import EpsilonGreedy
import matplotlib.pyplot as plt
from collections import deque
import keras
import tensorflow as tf
from keras.api.layers import Embedding, Flatten, Dense

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


class TaxiAgent(ABC):
    def __init__(self) -> None:
        self.metrics = AgentMetrics()
        self._env = gym.make("Taxi-v3") # Needed for subclasses to initialize.

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
                action = self._get_action(state, is_training)

                next_state, reward, terminated, truncated, _ = self._env.step(action)

                if is_training:
                    self._update(state, action, reward.__float__(), next_state, terminated or truncated)

                state = next_state
                episode_reward += reward.__float__()
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

    def __del__(self) -> None:
        try:
            self._env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")


class BasicQAgent(TaxiAgent):
    def __init__(self, epsilon: EpsilonGreedy, learning_rate: float, discount_factor: float) -> None:
        super().__init__()

        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._discount = discount_factor
        self._q_table = np.zeros((self._env.observation_space.n, self._env.action_space.n)) # type: ignore

    def _load(self) -> None:
        assert os.path.exists("static/weights/q_table.npy"), "Weights not found!"
        with open("static/weights/q_table.npy", "rb") as f:
            self._q_table = np.load(f)

    def _save(self) -> None:
        if not os.path.exists("static/weights"):
            os.makedirs("static/weights", exist_ok=True)

        with open("static/weights/q_table.npy", "wb") as f:
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
        assert os.path.exists("static/weights/sarsa_q_table.npy"), "Weights not found!"
        with open("static/weights/sarsa_q_table.npy", "rb") as f:
            self._q_table = np.load(f)

    def _save(self) -> None:
        if not os.path.exists("static/weights"):
            os.makedirs("static/weights", exist_ok=True)

        with open("static/weights/sarsa_q_table.npy", "wb") as f:
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
    def __init__(self, epsilon: EpsilonGreedy,
                 learning_rate: float,
                 discount_factor: float,
                 batch_size: int,
                 memory_size: int) -> None:
        super().__init__()

        self._replay = deque(maxlen=memory_size)
        self._s_batch = batch_size

        self._epsilon = epsilon
        self._discount = discount_factor

        self.__step_counter = 0

        self._policy = self.__create_model(self._env.observation_space.n, self._env.action_space.n) # type: ignore
        self._target = self.__create_model(self._env.observation_space.n, self._env.action_space.n) # type: ignore
        self._target.set_weights(self._policy.get_weights()) # sync

        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self._loss_fn = keras.losses.Huber()

        self._loss_history_episode = []

    def _load(self) -> None:
        assert os.path.exists("static/weights/policy.keras"), "Weights not found!"
        return keras.models.load_model('static/weights/policy.keras')  # type: ignore

    def _save(self) -> None:
        if not os.path.exists("static/weights"):
            os.makedirs("static/weights", exist_ok=True)

        self._policy.save('static/weights/policy.keras')  # type: ignore

    def __create_model(self, n_states: int, n_actions: int) -> keras.Model:
        model = keras.Sequential([
            Embedding(n_states, 128, input_length=1),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(n_actions, activation='linear')
        ])
        return model

    def _get_action(self, state: int, is_training: bool) -> int:
        if self._epsilon.should_be_random and is_training:
            return self._env.action_space.sample()

        q_values = self._policy(np.array([state], dtype=np.float32))
        return np.argmax(q_values[0]).item()
    
    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        self._replay.append((state, action, reward, next_state, terminated))

        if len(self._replay) < self._s_batch:
            # Not enough samples to train
            return

        loss = self.__train()

        if terminated:
            self._epsilon.update() # Saves epsilon history too
            # Save loss and epsilon
            self._loss_history_episode.append(loss)

        self.__step_counter += 1
        if self.__step_counter % 1000 == 0:
            self._target.set_weights(self._policy.get_weights())

    def __train(self) -> float:
        if len(self._replay) < self._s_batch:
            return -np.inf

        minibatch = random.sample(self._replay, self._s_batch)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*minibatch))

        actions_one_hot = tf.one_hot(actions, self._env.action_space.n) # type: ignore

        loss = self.train_steps(states, actions_one_hot,
                                rewards, next_states, dones)

        return loss # type: ignore

    @tf.function
    def train_steps(self, states, actions, rewards, next_states, dones):
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        next_q_values = self._target(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + self._discount * \
            max_next_q_values * (1 - dones)  # type: ignore

        with tf.GradientTape() as tape:
            q_values = self._policy(states)
            action_q_values = tf.reduce_sum(q_values * actions, axis=1)
            loss = self._loss_fn(targets, action_q_values)

        gradients = tape.gradient(loss, self._policy.trainable_variables)
        self._optimizer.apply_gradients(
            zip(gradients, self._policy.trainable_variables))  # type: ignore

        return loss


if __name__ == "__main__":
    basic = BasicQAgent(EpsilonGreedy(1.0, 0.999, 0.00), 0.1, 0.95)
    metrics = basic.train(10000, 1000)
    metrics.plot('static/metrics/basic.png')
    basic.record_video(3, 'static/movies/basic')

    sarsa = SarsaAgent(EpsilonGreedy(1.0, 0.999, 0.00), 0.1, 0.95)
    metrics = sarsa.train(10000, 1000)
    metrics.plot('static/metrics/sarsa.png')
    sarsa.record_video(3, 'static/movies/sarsa')

    dql = DeepQAgent(EpsilonGreedy(1.0, 0.998, 0.00), 0.0001, 0.95, 258, 25000)
    metrics = dql.train(2500, 1000)
    metrics.plot('static/metrics/dql.png')
    dql.record_video(3, 'static/movies/dql')
