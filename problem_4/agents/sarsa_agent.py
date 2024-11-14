import os
import numpy as np
from utils import EpsilonGreedy
from .abstraction import TaxiAgent
from typing import Tuple


class SarsaAgent(TaxiAgent):
    def __init__(self, epsilon: EpsilonGreedy, learning_rate: float, discount_factor: float) -> None:
        super().__init__(epsilon)

        self._learning_rate = learning_rate
        self._discount = discount_factor
        self._q_table = np.zeros(
            (self._env.observation_space.n, self._env.action_space.n))  # type: ignore

    def _load(self) -> None:
        assert os.path.exists(
            "static/weights/sarsa_q_table.npy"), "Weights not found!"
        with open("static/weights/sarsa_q_table.npy", "rb") as f:
            self._q_table = np.load(f)

    def _save(self) -> None:
        if not os.path.exists("static/weights"):
            os.makedirs("static/weights", exist_ok=True)

        with open("static/weights/sarsa_q_table.npy", "wb") as f:
            np.save(f, self._q_table)

    def _get_action(self, state: int) -> int:
        return np.argmax(self._q_table[state]).item()

    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        """
        Formula: Q(s, a) <- Q(s, a) + α[R(s) + γ Q(s', a') - Q(s, a)]
        """
        if terminated:
            target = reward
        else:
            next_action = super()._training_action(next_state, persist=True)
            target = reward + self._discount * \
                self._q_table[next_state, next_action]  # Q(s', a')

        self._q_table[state, action] += self._learning_rate * \
            (target - self._q_table[state, action])


if __name__ == "__main__":
    sarsa = SarsaAgent(EpsilonGreedy(1.0, 0.999, 0.00), 0.1, 0.95)
    metrics = sarsa.train(10000, 1000)
    metrics.plot('static/metrics/sarsa.png')
    sarsa.record_video(3, 'static/movies/sarsa')
