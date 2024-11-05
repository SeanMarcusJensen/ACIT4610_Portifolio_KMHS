if __name__ == '__main__':
    import agents
    from utils import EpsilonGreedy

    random_policy = agents.RandomPolicyAgent()
    metrics = random_policy.train(10000, 1000)
    metrics.plot(None)

    heuristic_policy = agents.HeuristicPolicyAgent()
    metrics = heuristic_policy.train(10000, 1000)
    metrics.plot(None)
    heuristic_policy.watch(2, 30)

    basic = agents.BasicQAgent(
            EpsilonGreedy(1.0, 0.999, 0.00),
            0.1, 0.95)
    metrics = basic.train(10000, 1000)
    metrics.plot('static/metrics/basic.png')
    basic.record_video(3, 'static/movies/basic')

    sarsa = agents.SarsaAgent(
            EpsilonGreedy(1.0, 0.999, 0.00),
            0.1, 0.95)
    metrics = sarsa.train(10000, 1000)
    metrics.plot('static/metrics/sarsa.png')
    sarsa.record_video(3, 'static/movies/sarsa')

    dql = agents.DeepQAgent(
            EpsilonGreedy(1.0, 0.998, 0.00),
            0.0001, 0.95, 258, 25000)
    metrics = dql.train(2500, 1000)
    metrics.plot('static/metrics/dql.png')
    dql.record_video(3, 'static/movies/dql')
