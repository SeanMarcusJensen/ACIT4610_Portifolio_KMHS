
from gym.spaces import Space
import pickle
import numpy as np
from gym import Env
import matplotlib.pyplot as plt
from typing import Tuple, List
from numpy.typing import NDArray
import keras
import tensorflow as tf
from .abstraction import Agent


class ExperienceReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer: List[Tuple[int, int, float, int, bool]] = []

    def add(self, state: int, action: int, reward: float, next_state: int, terminal: bool):
        self.buffer.append((state, action, reward, next_state, terminal))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Tuple[int, int, float, int, bool]]:
        indices = np.random.randint(0, len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


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

    def __init__(self, EPSILON: float = 1.0, EPSILON_DECAY: float = 0.996, LEARNING_RATE: float = 0.1, DISCOUNT_FACTOR: float = 0.9, EPSILON_MIN: float = 0.01, BATCH_SIZE: int = 1000):
        self.EPSILON = EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.EPSILON_MIN = EPSILON_MIN
        self.BATCH_SIZE = BATCH_SIZE

        self.__sync_after_steps = 1001

        self.__current_reward: float = 0.0
        self.__is_training = True

        self.__policy: keras.Model | None = None
        self.__target: keras.Model | None = None

        self.__loss_fn = keras.losses.MeanSquaredError()

        self.__encoded_states: NDArray[np.float64] | None = None

        # Buffer size should be a parameter.
        self.__replay_buffer = ExperienceReplayBuffer(2000)

        self.__current_steps = 0
        self.__current_episode = 0

        self.__rewards: NDArray[np.float64] | None = None

        self.__random_actions = 0
        self.__policy_actions = 0

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

        try:
            if self.__is_training and (np.random.uniform(0, 1) < self.EPSILON):
                action = action_space.sample()
                self.__random_actions += 1
            else:
                self.__policy_actions += 1
                policy_input = self.__encoded_states[state].reshape(1, -1)
                print(f"Input Policy: {policy_input}")
                # Dont know if reshape is good here.
                policy_output = self.__policy(policy_input)
                print(f"Policy Output: {policy_output}")
                q_values = np.argmax(policy_output).item()
                print(f"Action to take: {q_values}")
                action = q_values
            return action
        except Exception as e:
            print("Error in find_action, random action is taken.", e)
            return action_space.sample()

    def update(self, state: int, action: int, reward: float, next_state: int, terminal: bool) -> None:
        self.__replay_buffer.add(state, action, reward, next_state, terminal)
        self.__current_reward += reward
        self.__current_steps += 1

    def end_of_episode(self) -> None:
        assert self.__replay_buffer is not None, "Replay buffer is not initialized."
        assert self.__policy is not None, "Policy model is not initialized."
        assert self.__target is not None, "Target model is not initialized."
        assert self.__rewards is not None, "Rewards array is not initialized."

        print(
            f"End of episode: {self.__current_episode}, reward: {self.__current_reward}")

        self.__rewards[self.__current_episode] = self.__current_reward
        self.__current_reward = 0.0

        has_enough_data = len(self.__replay_buffer) > self.BATCH_SIZE
        has_rewards = True  # np.sum(self.__rewards) > 0
        print(
            f"Current Episode: {self.__current_episode}, has_enough_data: {has_enough_data}, has_rewards: {has_rewards}")
        if self.__is_training and has_enough_data and has_rewards:
            print("Training the model.")
            batch = self.__replay_buffer.sample(self.BATCH_SIZE)
            self.__optimize_model(batch)

            self.EPSILON = max(
                self.EPSILON_MIN, self.EPSILON_DECAY * self.EPSILON)

            if self.__current_steps >= self.__sync_after_steps:
                self.__target.set_weights(self.__policy.get_weights())
                self.__current_steps = 0

        self.__current_episode += 1

    def __optimize_model(self, batch: List[Tuple[int, int, float, int, bool]]) -> None:
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
                target = [reward]
            else:
                print(f"Next State: {next_state}")
                input_state = tf.constant(
                    self.__encoded_states[next_state].reshape(1, -1))

                prediction = self.__target(input_state)

                target = reward + self.DISCOUNT_FACTOR * np.argmax(prediction)

            current_q = self.__policy(
                self.__encoded_states[state].reshape(1, -1)).numpy()
            print(f"Current Q: {current_q}")
            q_list.append(current_q[0])

            target_prediction = self.__target(
                self.__encoded_states[next_state].reshape(1, -1))
            print(f"Target Prediction: {target_prediction}")
            target_q = target_prediction.numpy()[0]
            print(f"Target Q: {target_q}")
            target_q[action] = target
            target_q_list.append(target_q)

        self.__loss_fn(q_list, target_q_list)

        # TODO: Fit model.
        input_list = tf.constant(tf.stack(q_list))
        target_list = tf.constant(tf.stack(target_q_list))
        print(f"Input List: {input_list}")
        print(f"Target List: {target_list}")
        self.__policy.fit(input_list, target_list, epochs=1)

    def plot_rewards(self) -> None:
        # TODO:
        print(f"Random Actions: {self.__random_actions}")
        print(f"Policy Actions: {self.__policy_actions}")

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
                keras.layers.Dense(action_space)
            ])
        else:
            self.__policy = keras.models.load_model(
                'deep_q_learning.keras')  # type: ignore
        assert self.__policy is not None, "Policy model is not initialized."
        self.__policy.compile(
            optimizer='adam', loss=self.__loss_fn, metrics=['accuracy'])

        self.__target = keras.models.clone_model(self.__policy)
