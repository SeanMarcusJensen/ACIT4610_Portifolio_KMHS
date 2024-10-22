from .abstraction import Agent

import gym


class Taxi:
    @staticmethod
    def run(agent: Agent, n_episodes: int, max_steps: int | None, is_training: bool = True) -> None:
        env = gym.make(
            'Taxi-v3', render_mode='human' if not is_training else None)

        try:
            agent.initialize(
                env.action_space, env.observation_space, n_episodes, is_training=is_training)

            for _ in range(n_episodes):
                state = env.reset()[0]
                truncated = False
                terminated = False

                current_step = 0
                while not truncated and not terminated:
                    current_step += 1
                    action = agent.find_action(state, env.action_space)
                    next_state, reward, terminated, truncated, _ = env.step(
                        action)
                    agent.update(state, action, reward, next_state, terminated)
                    state = next_state

                    if truncated or terminated:
                        break

                    if max_steps is not None:
                        if current_step > max_steps:
                            break

                agent.end_of_episode()

            if is_training:
                agent.save()
        except Exception as e:
            print(f"Error in running the agent: {e}")

        env.close()
