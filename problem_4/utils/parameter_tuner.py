

def find_best_parameters(callable, n_trials=200, plot=False):
    import optuna
    import numpy as np
    from utils.epsilon_greedy import EpsilonGreedy
    from utils.agent_metrics import AgentMetrics
    from typing import List

    tune_metrics = {
            'epsilon_decay': [],
            'learning_rate': [],
            'discount_factor': [],
            'score_min': [],
            'score_avg': [],
            'score_max': [],
            'step_min': [],
            'step_avg': [],
            'step_max': [],
            'time_min': [],
            'time_avg': [],
            'time_max': [],
            }

    metric_list: List[AgentMetrics] = []

    def create_objective(callable):

        def objective(trial):
            epsilon_decay = trial.suggest_float('epsilon_decay', 0.8, 0.999)
            learning_rate = trial.suggest_float('learning_rate', 0.1, 0.8)
            discount_factor = trial.suggest_float('discount_factor', 0.1, 0.99)

            epsilon = EpsilonGreedy(1.0, epsilon_decay, 0.0)
            agent = callable(epsilon, learning_rate, discount_factor)

            metrics = agent.train(n_episodes=1500, step_limit_per_episode=400)
            metric_list.append(metrics)

            score = np.mean(metrics.episode_rewards)

            tune_metrics['epsilon_decay'].append(epsilon_decay)
            tune_metrics['learning_rate'].append(learning_rate)
            tune_metrics['discount_factor'].append(discount_factor)

            return score

        return objective

    study = optuna.create_study(direction='maximize')
    study.optimize(create_objective(callable), n_trials=n_trials)

    for metric in metric_list:
        reward = np.array(metric.episode_rewards)
        time = np.array(metric.episode_time)
        steps = np.array(metric.episode_steps)
        print(f"Length of reward: {len(reward)}")
        print(f"Length of time: {len(time)}")
        print(f"Length of steps: {len(steps)}")

        tune_metrics['score_min'].append(reward.min())
        tune_metrics['score_avg'].append(reward.mean())
        tune_metrics['score_max'].append(reward.max())
        tune_metrics['step_min'].append(steps.min())
        tune_metrics['step_avg'].append(steps.mean())
        tune_metrics['step_max'].append(steps.max())
        tune_metrics['time_min'].append(time.min())
        tune_metrics['time_avg'].append(time.mean())
        tune_metrics['time_max'].append(time.max())

    if plot:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()

    return study.best_params, tune_metrics


if __name__ == '__main__':
    from agents import BasicQAgent, SarsaAgent
    from utils import EpsilonGreedy
    import pandas as pd
        
    study_q, q_metrics = find_best_parameters(lambda x, y, z: BasicQAgent(x, y, z))
    study_s, s_metrics  = find_best_parameters(lambda x, y, z: SarsaAgent(x, y, z))

    agent = BasicQAgent(EpsilonGreedy(1.0, study_q['epsilon_decay'], 0.0),
                        study_q['learning_rate'], study_q['discount_factor'])
    metrics = agent.train(n_episodes=1500, step_limit_per_episode=400)
    metrics.save('static/metrics/basic.csv')

    agent = SarsaAgent(EpsilonGreedy(1.0, study_s['epsilon_decay'], 0.0),
                        study_s['learning_rate'], study_s['discount_factor'])
    metrics = agent.train(n_episodes=1500, step_limit_per_episode=400)
    metrics.save('static/metrics/sarsa.csv')

    q_metrics = pd.DataFrame(q_metrics)
    s_metrics = pd.DataFrame(s_metrics)

    q_metrics.to_csv('static/metrics/basic_tune_metrics.csv', index=False)
    s_metrics.to_csv('static/metrics/sarsa_tune_metrics.csv', index=False)
