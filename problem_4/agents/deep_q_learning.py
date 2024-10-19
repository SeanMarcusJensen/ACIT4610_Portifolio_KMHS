
from gym.spaces import Space
import pickle
import numpy as np
from gym import Env
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import keras
import tensorflow as tf


class DeepQLearningAgent:
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

    def __init__(self):
        pass

    def __create_model(self, observation_space: int, action_space: int) -> keras.Model:
        """ Create a neural network model for the Deep Q-Learning Agent.

        Args:
            observation_space (int): The number of states in the environment.
            action_space (int): The number of actions in the environment.

        Returns:
            keras.Model: Return a neural network model for the Deep Q-Learning Agent.
        """
        return keras.Sequential([
            keras.layers.Input(shape=(observation_space,)),
            keras.layers.Dense(action_space * 8, activation='relu'),
            keras.layers.Dense(action_space, activation='linear')
        ])

    def __one_hot_encode(self, state: int, observation_space: int) -> NDArray[np.float64]:
        """ One Hot Encode the state to become input for the neural network.

        Args:
            state (int): The current state of the environment.
            observation_space (int): The number of states in the environment.

        Returns:
            NDArray[np.float64]: Return a one-hot encoded state.
        """
        return np.eye(observation_space)[state]

    def __select_action(self, policy: keras.Model, state: int, observation_space: int) -> int:
        encoded_state = self.__one_hot_encode(state, observation_space)
        q_values = policy.predict(encoded_state)
        return np.argmax(q_values).item()

    def train(self, env: Env, N_EPISODES: int, LEARNING_RATE: float, DISCOUNT_FACTOR: float, EPSILON: float, EPSILON_DECAY: float, EPSILON_MIN: float):
        observation_space = int(env.observation_space.n)
        action_space = int(env.action_space.n)  # Number of states (6)

        policy = self.__create_model(observation_space, action_space)
        target = keras.models.clone_model(policy)

        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        loss_fn = keras.losses.MeanSquaredError()

        rewards = np.zeros(N_EPISODES)

        steps_after_last_sync = 0
        for episode in range(N_EPISODES):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while (not terminated) and (not truncated):
                steps_after_last_sync += 1
                # Choose an action
                # Epsilon-Greedy Policy
                if np.random.uniform(0, 1) < EPSILON:
                    action = env.action_space.sample()
                else:
                    action = self.__select_action(
                        policy, state, observation_space)

                # Take the action
                next_state, reward, terminated, truncated, _ = env.step(action)

                # TODO: Implement Experience Replay Buffer.

                state = next_state

            rewards[episode] = reward

            # TODO: Optimize models.
            # TODO: Implement Experience Replay Buffer.
            # TODO: Only train the model if the reward is positive and we have enough samples in the buffer.
            sum_reward = np.sum(rewards)
            if sum_reward > 0:
                print("Reward: ", sum_reward)

                # TODO: Get Sample from Experience Replay Buffer.

                # TODO: Train Model

                # Decay the epsilon value
                EPSILON = max(EPSILON_MIN, EPSILON_DECAY * EPSILON)

            if steps_after_last_sync >= 100:  # Should add as param
                # (SYNC) Update the target network.
                target.set_weights(policy.get_weights())
                steps_after_last_sync = 0

            # Update the Q-Table
            with tf.GradientTape() as tape:
                q_values = policy(self.__one_hot_encode(
                    state, observation_space))
                target_q_values = target(
                    self.__one_hot_encode(next_state, observation_space))

                q_value = q_values[0][action]
                target_q_value = reward + DISCOUNT_FACTOR * \
                    np.max(target_q_values)

                loss = loss_fn(q_value, target_q_value)

            grads = tape.gradient(loss, policy.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, policy.trainable_variables))
