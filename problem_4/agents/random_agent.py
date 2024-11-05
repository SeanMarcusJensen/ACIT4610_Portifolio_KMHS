from .abstraction import TaxiAgent
from utils import EpsilonGreedy

class RandomPolicyAgent(TaxiAgent):
    def __init__(self) -> None:
        super().__init__(EpsilonGreedy(1.0, 0.0, 1.0))

    def _load(self) -> None:
        return
    
    def _save(self) -> None:
        return

    def _get_action(self, state: int) -> int:
        """ Returns a random action. """
        return self._env.action_space.sample()
    
    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        """ No training is needed, because the agent is deterministic. """
        return

if __name__ == "__main__":
    random_policy = RandomPolicyAgent()
    metrics = random_policy.train(10000, 1000)
    metrics.plot(None)
