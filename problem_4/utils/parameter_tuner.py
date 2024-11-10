def find_best_parameters(callable, n_trials=200, plot=False):
    import optuna
    import numpy as np
    from utils.epsilon_greedy import EpsilonGreedy

    def create_objective(callable):
        def objective(trial):
            epsilon_decay = trial.suggest_float('epsilon_decay', 0.8, 0.999)
            learning_rate = trial.suggest_float('learning_rate', 0.1, 0.8)
            discount_factor = trial.suggest_float('discount_factor', 0.1, 0.99)

            epsilon = EpsilonGreedy(1.0, epsilon_decay, 0.0)
            agent = callable(epsilon, learning_rate, discount_factor)

            metrics = agent.train(n_episodes=1500, step_limit_per_episode=400)

            return np.mean(metrics.episode_rewards)
        return objective

    study = optuna.create_study(direction='maximize')
    study.optimize(create_objective(callable), n_trials=n_trials)
    if plot:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
    return study.best_params


if __name__ == '__main__':
    from utils.agent_metrics_comparer import AgentMetricsComparer
    from agents import BasicQAgent, SarsaAgent
    from utils import EpsilonGreedy
        
    study_q = find_best_parameters(lambda x, y, z: BasicQAgent(x, y, z))
    study_s = find_best_parameters(lambda x, y, z: SarsaAgent(x, y, z))

    agent = BasicQAgent(EpsilonGreedy(1.0, study_q['epsilon_decay'], 0.0),
                        study_q['learning_rate'], study_q['discount_factor'])
    metrics = agent.train(n_episodes=1500, step_limit_per_episode=400)
    AgentMetricsComparer.plot(metrics.as_pandas())

    agent = SarsaAgent(EpsilonGreedy(1.0, study_s['epsilon_decay'], 0.0),
                        study_s['learning_rate'], study_s['discount_factor'])
    metrics = agent.train(n_episodes=1500, step_limit_per_episode=400)
    AgentMetricsComparer.plot(metrics.as_pandas())

