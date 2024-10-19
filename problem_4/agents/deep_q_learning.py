
from gym.spaces import Space
import pickle
import numpy as np
from gym import Env
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import keras
import tensorflow as tf
from .abstraction import Agent


class ExperienceReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state: int, action: int, reward: float, next_state: int, terminal: bool):
        self.buffer.append((state, action, reward, next_state, terminal))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size: int):
        return np.random.choice(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def __iter__(self):
        return iter(self.buffer)

    def __str__(self):
        return str(self.buffer)

    def __repr__(self):
        return repr(self.buffer)


class DeepQLearningAgent(Agent):
    """ Deep Q-Learning Agent.

    Setup:
        Input nodes = 500 (observation_space),
        Output nodes = 6 (action_space),
        Dense = 1 layer with 500 nodes, ( Fully Connected Layer ).

    Steps:
        1. Initialize the Q-Table with zeros.
        ...
        n. One Hot encode the state to become input for the neural network.
    """

    def __init__(self, EPSILON: float = 1.0, EPSILON_DECAY: float = 0.001, LEARNING_RATE: float = 0.1, DISCOUNT_FACTOR: float = 0.9, EPSILON_MIN: float = 0.01, BATCH_SIZE: int = 100):
        self.EPSILON = EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.EPSILON_MIN = EPSILON_MIN
        self.BATCH_SIZE = BATCH_SIZE

        self.__sync_after_steps = 100

        self.__current_reward = 0
        self.__is_training = True

        self.__policy: keras.Model | None = None
        self.__target: keras.Model | None = None

        self.__loss_fn = keras.losses.MeanSquaredError()

        self.__encoded_states: NDArray[np.float64] | None = None

        # Buffer size should be a parameter.
        self.__replay_buffer = ExperienceReplayBuffer(1000)

        self.__current_steps = 0
        self.__current_episode = 0

        self.__rewards: NDArray[np.float64] | None = None

    def save(self) -> None:
        assert self.__policy is not None, "Policy model is not initialized."
        self.__policy.save('deep_q_learning.keras')

    def initialize(self, action_space: Space, observation_space: Space, n_episodes: int, is_training: bool) -> None:
        self.__is_training = is_training
        self.__create_model(observation_space.n, action_space.n)
        self.__encoded_states = np.eye(observation_space.n)
        self.__rewards = np.zeros(n_episodes)

    def find_action(self, state: int, action_space: Space) -> int:
        assert self.__policy is not None, "Policy model is not initialized."
        assert self.__encoded_states is not None, "Encoded states are not initialized."

        action: int = 0

        if self.__is_training and (np.random.uniform(0, 1) < self.EPSILON):
            action = action_space.sample()
        else:
            q_values = np.argmax(self.__policy(
                self.__encoded_states[state].reshape(1, -1))).item()  # Dont know if reshape is good here.
            action = q_values

        assert action is not None, "Action is not initialized."
        assert 0 <= action <= 5, "Action is less than 0."
        return action

    def update(self, state: int, action: int, reward: float, next_state: int, terminal: bool) -> None:
        self.__replay_buffer.add(state, action, reward, next_state, terminal)
        self.__current_reward += reward
        self.__current_steps += 1
        self.__current_episode += 1

    def end_of_episode(self) -> None:
        assert self.__replay_buffer is not None, "Replay buffer is not initialized."
        assert self.__policy is not None, "Policy model is not initialized."
        assert self.__target is not None, "Target model is not initialized."
        assert self.__rewards is not None, "Rewards array is not initialized."

        if self.__current_reward <= 0:
            return

        self.__rewards[self.__current_episode] = self.__current_reward

        if self.__is_training and len(self.__replay_buffer) > self.BATCH_SIZE and np.sum(self.__rewards) > 0:
            batch = self.__replay_buffer.sample(self.BATCH_SIZE)
            self.__optimize_model(batch)

            self.EPSILON = max(
                self.EPSILON_MIN, self.EPSILON_DECAY * self.EPSILON)

            if self.__current_steps >= self.__sync_after_steps:
                self.__target.set_weights(self.__policy.get_weights())
                self.__current_steps = 0

        self.__current_reward = 0

    def __optimize_model(self, batch: np.ndarray) -> None:
        assert self.__policy is not None, "Policy model is not initialized."
        assert self.__target is not None, "Target model is not initialized."
        assert self.__encoded_states is not None, "Encoded states are not initialized."

        q_list = []
        target_q_list = []
        # Update the Q-Table
        # Iterate over the batch
        # Collect the Q-Values for the state
        for state, action, reward, next_state, terminal in batch:
            if terminal:
                target = reward
            else:
                target = reward + self.DISCOUNT_FACTOR * \
                    np.max(self.__target.predict(
                        self.__encoded_states[next_state]))

            current_q = self.__policy(self.__encoded_states[state])
            q_list.append(current_q)

            target_q = self.__target(self.__encoded_states[next_state])
            target_q[action] = target
            target_q_list.append(target_q)

        # TODO: Fit model.
        self.__policy.fit(self.__encoded_states, target_q_list)

    def plot_rewards(self) -> None:
        # TODO:
        pass

    def __create_model(self, observation_space: int, action_space: int) -> None:
        """ Create a neural network model for the Deep Q-Learning Agent.

        Args:
            observation_space (int): The number of states in the environment.
            action_space (int): The number of actions in the environment.

        Returns:
            keras.Model: Return a neural network model for the Deep Q-Learning Agent.
        """
        if self.__is_training:
            self.__policy = keras.Sequential([
                keras.layers.Input(shape=(observation_space,)),
                keras.layers.Dense(action_space * 8, activation='relu'),
                keras.layers.Dense(action_space, activation='linear')
            ])
        else:
            self.__policy = keras.models.load_model(
                'deep_q_learning.keras')  # type: ignore
        assert self.__policy is not None, "Policy model is not initialized."
        self.__policy.compile(
            optimizer='adam', loss=self.__loss_fn, metrics=['accuracy'])

        self.__target = keras.models.clone_model(self.__policy)
