import os
import keras
import random
import numpy as np
import tensorflow as tf
from collections import deque
from utils import EpsilonGreedy
from .abstraction import TaxiAgent
from keras.api.layers import Embedding, Flatten, Dense


class DeepQAgent(TaxiAgent):
    def __init__(self,
                 epsilon: EpsilonGreedy,
                 learning_rate: float,
                 discount_factor: float,
                 batch_size: int,
                 memory_size: int) -> None:
        super().__init__(epsilon)

        self._replay = deque(maxlen=memory_size)
        self._s_batch = batch_size

        self._discount = discount_factor

        self.__step_counter = 0

        self._policy: keras.Model = self.__create_model(
            self._env.observation_space.n, self._env.action_space.n)  # type: ignore
        self._target = self.__create_model(
            self._env.observation_space.n, self._env.action_space.n)  # type: ignore
        self._target.set_weights(self._policy.get_weights())  # sync

        self._optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self._loss_fn = keras.losses.Huber()

        self._loss_history_episode = []

    def _load(self) -> None:
        assert os.path.exists(
            "static/weights/policy.keras"), "Weights not found!"

        self._policy = keras.models.load_model(  # type: ignore
            'static/weights/policy.keras', compile=False)

    def _save(self) -> None:
        if not os.path.exists("static/weights"):
            os.makedirs("static/weights", exist_ok=True)

        self._policy.save('static/weights/policy.keras')  # type: ignore

    def __create_model(self, n_states: int, n_actions: int) -> keras.Model:
        model = keras.Sequential([
            Embedding(n_states, 128),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(n_actions, activation='linear')
        ])

        return model

    def _get_action(self, state: int) -> int:
        q_values = self._policy(np.array([state], dtype=np.float32))
        return np.argmax(q_values).item()

    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        self._replay.append((state, action, reward, next_state, terminated))

        if len(self._replay) < self._s_batch:
            # Not enough samples to train
            return

        loss = self.__train()

        if terminated:
            # Save loss
            self._loss_history_episode.append(loss)

        self.__step_counter += 1
        if self.__step_counter % 1000 == 0:
            self._target.set_weights(self._policy.get_weights())

    def __train(self) -> float:
        minibatch = random.sample(self._replay, self._s_batch)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*minibatch))

        actions_one_hot = tf.one_hot(
            actions, self._env.action_space.n)  # type: ignore

        loss = self.train_steps(states, actions_one_hot,
                                rewards, next_states, dones)

        return loss  # type: ignore

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
    from utils import create_progress_tracker
    dql = DeepQAgent(EpsilonGreedy(1.0, 0.998, 0.00), 0.0001, 0.95, 258, 25000)
    metrics = dql.train(
        200, 200, create_progress_tracker("Deep Q-Learning Agent"))
    dql.record_video(3, 'static/movies/dql')
