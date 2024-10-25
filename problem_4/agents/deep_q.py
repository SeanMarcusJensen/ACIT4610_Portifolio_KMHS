
import keras
import random
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
from gym.spaces import Space
from collections import deque
from .abstraction import Agent
from typing import Tuple, List
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from keras.api.layers import Dense, Embedding, Flatten


@dataclass
class EpsilonGreedy:
    """ Epsilon Greedy Algorithm """
    value: float
    decay: float
    min: float
    history: List[float] = field(default_factory=list)

    @property
    def should_be_random(self) -> bool:
        return np.random.rand() < self.value

    def update(self, episode: int):
        self.value = max(self.min, self.value * self.decay)
        self.history.append(self.value)


@ dataclass
class Metrics:
    loss_: float = 0.0
    rewards_: float = 0.0
    current_episode: int = 0
    current_steps: int = 0
    loss: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    def step(self, loss: float, reward: float):
        self.current_steps += 1
        self.loss_ += loss
        self.rewards_ += reward

    def episode(self):
        self.loss.append(self.loss_)
        self.rewards.append(self.rewards_)
        self.steps.append(self.current_steps)
        self.current_steps = 0
        self.current_episode += 1
        self.rewards_ = 0.0
        self.loss_ = 0.0

    def log(self):
        print(
            f"Episode: {self.current_episode} Loss: {self.loss[-1]}, Reward: {self.rewards[-1]}")


@ dataclass
class ActionCache:
    max_repeated = 7
    __cache = deque(maxlen=max_repeated)

    def append(self, action: int):
        self.__cache.append(action)

    def is_repeated(self) -> bool:
        return len(np.unique(self.__cache)) <= 1


"""
Issues Met:
- The DQL model found out that standing in the wall
    and not moving is the best strategy to get the highest reward...

"""


class DeepQAgent(Agent):
    def __init__(self,
                 batch_size: int = 64,
                 lr: float = 0.01,
                 discount: float = 0.80,
                 epsilon: EpsilonGreedy = EpsilonGreedy(1.0, 0.95, 0.05),
                 ):
        self.__metrics = Metrics()
        self.__action_cache = ActionCache()

        self.__batch_size = batch_size
        self.__buffer = deque([], maxlen=50000)

        self.__discount = discount
        self.__epsilon = epsilon

        self.training = True

        self.__optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.__loss_fn = keras.losses.Huber()

        self.__policy: keras.Model
        self.__target: keras.Model

        self.__step_counter = 0
        self.__n_states = 0
        self.__n_actions = 0
        self.__current_reward: float = 0.0

    def initialize(self, action_space, observation_space, n_episodes, is_training=True):
        self.training = is_training
        self.__n_actions = action_space.n
        self.__n_states = observation_space.n
        self.__policy = self.create_model(
            self.__n_states, self.__n_actions, is_training)
        self.__target = self.create_model(
            self.__n_states, self.__n_actions, is_training)
        self.sync_target()

    def sync_target(self):
        self.__target.set_weights(self.__policy.get_weights())

    def create_model(self, n_states: int, n_actions: int, training: bool) -> keras.Model:
        if training:
            model = keras.Sequential([
                Embedding(n_states, 128, input_length=1),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(n_actions, activation='linear')
            ])
            return model

        return keras.models.load_model('policy.keras')  # type: ignore

    def find_action(self, state: int, action_space: Space) -> int:
        action: int = 0
        if self.training and self.__epsilon.should_be_random:
            action = action_space.sample()
        else:
            q_values = self.__policy(np.array([state], dtype=np.float32))
            action = np.argmax(q_values[0]).item()

        # Tool to prevent the agent from getting stuck in a loop
        # self.__action_cache.append(action)
        # if self.__action_cache.is_repeated():
        #     action = action_space.sample()

        return action

    def update(self, state: int, action: int, reward: float, next_state: int, terminated: bool):
        self.__buffer.append((state, action, reward, next_state, terminated))
        self.__current_reward += reward
        loss = self.train()
        self.__metrics.step(loss, reward)

    def end_of_episode(self) -> None:
        self.__metrics.episode()
        self.__metrics.log()
        self.__current_reward = 0.0

    def save(self) -> None:
        self.__policy.save('policy.keras')

    def train(self) -> float:
        if len(self.__buffer) < self.__batch_size:
            return -np.inf

        minibatch = random.sample(self.__buffer, self.__batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*minibatch))

        actions_one_hot = tf.one_hot(actions, self.__n_actions)

        loss = self.train_steps(states, actions_one_hot,
                                rewards, next_states, dones)

        self.__epsilon.update(self.__metrics.current_episode)

        self.__step_counter += 1
        if self.__step_counter % 1000 == 0:
            self.sync_target()
        return loss  # type: ignore

    @ tf.function
    def train_steps(self, states, actions, rewards, next_states, dones):
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        next_q_values = self.__target(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + self.__discount * \
            max_next_q_values * (1 - dones)  # type: ignore

        with tf.GradientTape() as tape:
            q_values = self.__policy(states)
            action_q_values = tf.reduce_sum(q_values * actions, axis=1)
            loss = self.__loss_fn(targets, action_q_values)

        gradients = tape.gradient(loss, self.__policy.trainable_variables)
        self.__optimizer.apply_gradients(
            zip(gradients, self.__policy.trainable_variables))  # type: ignore

        return loss

    def plot(self) -> None:
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

        plt.show()


if __name__ == "__main__":
    from .taxi import Taxi
    agent = DeepQAgent(
        batch_size=128,
        lr=0.001,
        discount=0.95,
        epsilon=EpsilonGreedy(1.0, 0.997, 0.05)
    )

    Taxi.run(agent, n_episodes=1000, steps_per_episode=1000, is_training=True)
    agent.plot()
    Taxi.run(agent, n_episodes=10, steps_per_episode=100, is_training=False)
