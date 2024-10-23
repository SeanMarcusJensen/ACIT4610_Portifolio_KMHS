from .abstraction import Agent

import gym


class Taxi:
    @staticmethod
    def run(agent: Agent, n_episodes: int, steps_per_episode: int = 1000, is_training: bool = True) -> None:
        env = gym.make(
            'Taxi-v3', render_mode='human' if not is_training else None)

        try:
            agent.initialize(
                env.action_space, env.observation_space, n_episodes, is_training=is_training)

            for _ in range(n_episodes):
                state = env.reset()[0]
                for _ in range(steps_per_episode):
                    action = agent.find_action(state, env.action_space)
                    next_state, reward, terminated, truncated, _ = env.step(
                        action)
                    agent.update(state, action, reward, next_state, terminated)
                    state = next_state
                    if truncated or terminated:
                        break

                agent.end_of_episode()

            if is_training:
                agent.save()
        except Exception as e:
            print(f"Error in running the agent: {e}")

        env.close()
