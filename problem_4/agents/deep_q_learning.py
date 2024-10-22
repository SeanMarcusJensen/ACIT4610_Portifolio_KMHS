
from gym.spaces import Space
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from numpy.typing import NDArray
import tensorflow as tf
import keras
from .abstraction import Agent

keras.config.disable_interactive_logging()


class ExperienceReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer: List[Tuple[int, int, float, int, bool]] = []

    def add(self, state: int, action: int, reward: float, next_state: int, terminal: bool):
        self.buffer.append((state, action, reward, next_state, terminal))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Tuple[int, int, float, int, bool]]:
        assert len(self.buffer) >= batch_size, "Not enough data in the buffer."
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

    def __init__(self, EPSILON: float = 1.0, EPSILON_DECAY: float = 0.95, LEARNING_RATE: float = 0.01, DISCOUNT_FACTOR: float = 0.95, EPSILON_MIN: float = 0.01, BATCH_SIZE: int = 64, MEMORY_SIZE: int = 200000, SYNC_AFTER_STEPS: int = 100000):
        self.EPSILON = EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.EPSILON_MIN = EPSILON_MIN
        self.BATCH_SIZE = BATCH_SIZE

        self.__sync_after_steps = SYNC_AFTER_STEPS

        self.__current_reward: float = 0.0
        self.__current_steps = 0
        self.__current_episode = 0

        self.__random_actions = 0
        self.__policy_actions = 0

        self.__is_training = True
        self.__initialized = False

        self.__rewards: NDArray[np.float64] = np.ndarray(0)
        # One hot encoded states.
        self.__encoded_states: NDArray[np.int8] = np.ndarray(0)

        self.__policy: keras.Model
        self.__target: keras.Model

        # Buffer size should be a parameter.
        self.__replay_buffer = ExperienceReplayBuffer(MEMORY_SIZE)

        self.__optimizer = keras.optimizers.Adam(
            learning_rate=self.LEARNING_RATE)
        self.__loss_fn = keras.losses.MeanSquaredError()
        self.__n_episodes = 0

    def save(self) -> None:
        self.__policy.save('deep_q_learning.keras')

    def initialize(self, action_space: Space, observation_space: Space, n_episodes: int, is_training: bool) -> None:
        self.__initialized = True
        self.__n_episodes = n_episodes
        self.__is_training = is_training
        self.__create_model(observation_space.n, action_space.n)
        self.__encoded_states = np.eye(observation_space.n, dtype=np.int8)
        self.__rewards = np.zeros(n_episodes)

    def find_action(self, state: int, action_space: Space) -> int:
        assert self.__initialized, "Agent is not initialized."

        action: int = 0

        try:
            if self.__is_training and (np.random.uniform(0, 1) < self.EPSILON):
                action = action_space.sample()
                self.__random_actions += 1
            else:
                self.__policy_actions += 1
                policy_input = self.__encoded_states[np.newaxis, state]
                policy_output = self.__policy(policy_input)
                q_values = np.argmax(policy_output[0]).item()
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
        assert self.__initialized, "Agent is not initialized."

        self.__rewards[self.__current_episode] = self.__current_reward
        self.__current_reward = 0.0

        has_enough_data = len(self.__replay_buffer) > self.BATCH_SIZE

        if self.__is_training and has_enough_data and self.__current_episode > 45:
            print(f"Episode: {self.__current_episode}/{self.__n_episodes}")
            batch = self.__replay_buffer.sample(self.BATCH_SIZE)
            self.__optimize_model(batch)

            self.EPSILON = max(
                self.EPSILON_MIN, self.EPSILON_DECAY * self.EPSILON)

            # if self.__current_steps >= self.__sync_after_steps:
            #     # Update the target model.
            #     self.__target.set_weights(self.__policy.get_weights())
            #     self.__current_steps = 0

        self.__current_episode += 1

    def __optimize_model(self, batch: List[Tuple[int, int, float, int, bool]]) -> None:
        assert self.__initialized, "Agent is not initialized."

        # Get the states, actions, rewards, next_states, and terminals from the batch.
        states, actions, rewards, next_states, terminals = zip(
            *batch, strict=True)

        # One hot encode the states.
        states = self.__encoded_states[np.array(states)]

        # One hot encode the next states.
        next_states = self.__encoded_states[np.array(next_states)]

        next_q_values = self.__policy.predict(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)

        terminals = np.array(terminals, dtype=np.int8)
        target_q_values = (rewards + (1 - terminals) *
                           self.DISCOUNT_FACTOR * max_next_q_values).reshape(-1, 1)

        # One hot encode the actions.
        encoded_actions = np.eye(self.__policy.output_shape[1])[
            np.array(actions)]

        # Train the model.
        with tf.GradientTape() as tape:
            q_values = self.__policy(states, training=True)
            q_values = tf.reduce_sum(q_values * encoded_actions, axis=1)
            loss = tf.reduce_mean(self.__loss_fn(target_q_values, q_values))

        gradients = tape.gradient(loss, self.__policy.trainable_variables)
        self.__optimizer.apply_gradients(
            zip(gradients, self.__policy.trainable_variables))  # type: ignore

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
                # Use softmax to get the probabilities of each action.
                keras.layers.Dense(action_space, activation='softmax')
            ])
        else:
            model = keras.models.load_model(
                'deep_q_learning.keras', compile=True)
            self.__policy = model  # type: ignore

        self.__target = keras.models.clone_model(self.__policy)


if __name__ == '__main__':
    from .taxi import Taxi
    agent = DeepQLearningAgent()
    Taxi.run(agent, 10, is_training=False)
    agent.plot_rewards()
