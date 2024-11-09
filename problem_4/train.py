import agents
from utils import EpsilonGreedy, create_progress_tracker


def train(EPISODES: int, R_EPS: int = 3) -> None:
    random_policy = agents.RandomPolicyAgent()
    metrics = random_policy.train(
        EPISODES, 200, create_progress_tracker('Random Policy Agent'))
    metrics.save('static/metrics/random.csv')
    random_policy.record_video(R_EPS, 'static/movies/random')

    heuristic_policy = agents.HeuristicPolicyAgent()
    metrics = heuristic_policy.train(
        EPISODES, 200, create_progress_tracker("Heuristic Policy Agent"))
    metrics.save('static/metrics/heuristic.csv')
    heuristic_policy.record_video(R_EPS, 'static/movies/heuristic')

    basic = agents.BasicQAgent(
        EpsilonGreedy(1.0, 0.999, 0.00),
        0.1, 0.95)
    metrics = basic.train(
        EPISODES, 200, create_progress_tracker("Basic Q-Learning Agent"))
    metrics.save('static/metrics/basic.csv')
    basic.record_video(R_EPS, 'static/movies/basic')

    sarsa = agents.SarsaAgent(
        EpsilonGreedy(1.0, 0.999, 0.00),
        0.1, 0.95)
    metrics = sarsa.train(
        EPISODES, 200, create_progress_tracker("Sarsa Agent"))
    metrics.save('static/metrics/sarsa.csv')
    sarsa.record_video(R_EPS, 'static/movies/sarsa')

    dql = agents.DeepQAgent(
        EpsilonGreedy(1.0, 0.998, 0.00),
        0.0001, 0.95, 258, 25000)
    metrics = dql.train(
        EPISODES, 200, create_progress_tracker("Deep Q-Learning Agent"))
    metrics.save('static/metrics/dql.csv')
    dql.record_video(R_EPS, 'static/movies/dql')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10_000,
                        help='Number of episodes to train the agent.')
    parser.add_argument('--record', type=int, default=3,
                        help='Number of episodes to record.')
    args = parser.parse_args()
    train(args.episodes, args.record)
