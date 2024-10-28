from gym import Space
from agents.abstraction import Agent


class SARSAAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, action_space: Space, observation_space: Space, n_episodes: int, is_training: bool) -> None:
        return super().initialize(action_space, observation_space, n_episodes, is_training)

    def find_action(self, state: int, action_space: Space) -> int:
        return super().find_action(state, action_space)

    def update(self, state: int, action: int, reward: float, next_state: int, terminal: bool) -> None:
        return super().update(state, action, reward, next_state, terminal)

    def end_of_episode(self) -> None:
        return super().end_of_episode()

    def save(self) -> None:
        return super().save()

    def plot(self, save_location: str | None = None) -> None:
        return


if __name__ == "__main__":
    from .taxi import Taxi
    agent = SARSAAgent(

    )

    Taxi.run(agent, n_episodes=100, steps_per_episode=1000, is_training=True)
    agent.plot()
    Taxi.run(agent, n_episodes=10, steps_per_episode=100, is_training=False)
