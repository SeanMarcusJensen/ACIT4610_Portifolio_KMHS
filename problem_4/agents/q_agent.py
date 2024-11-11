import os
import numpy as np
from utils import EpsilonGreedy
from .abstraction import TaxiAgent

class BasicQAgent(TaxiAgent):
    def __init__(self, epsilon: EpsilonGreedy, learning_rate: float, discount_factor: float) -> None:
        super().__init__(epsilon)

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

    def _get_action(self, state: int)-> int:
        return np.argmax(self._q_table[state]).item()

    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        """
        Formula: Q(a, s) <- Q(a, s) + α[R(s) + γ max_a' Q(a', s') - Q(a, s)]
        """
        if terminated:
            reward = 0

        self._q_table[state, action] += self._learning_rate * \
                (reward + self._discount * np.max(self._q_table[next_state]) - self._q_table[state, action])

if __name__ == "__main__":
    basic = BasicQAgent(EpsilonGreedy(1.0, 0.999, 0.00), 0.1, 0.95)
    metrics = basic.train(10000, 1000)
    metrics.plot('static/metrics/basic.png')
    basic.record_video(3, 'static/movies/basic')
