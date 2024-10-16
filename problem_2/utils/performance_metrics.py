import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

def analyze_returns(**kwargs):
    """Analyse and compare the expected return and the stability of the runs."""
    returns_analysis = []
    
    for name, data in kwargs.items():
        mean_return = data['returns'].mean()
        std_return = data['returns'].std()
        returns_analysis.append({
            'Algorithm': name,
            'Mean Return': mean_return,
            'Stability (Std Return)': std_return
        })
    
    returns_comparison_table = pd.DataFrame(returns_analysis)
    print('Returns comparison table:')
    display(returns_comparison_table)
    
    # Plot comparison of returns
    plt.figure(figsize=(12, 6))
    for name, data in kwargs.items():
        plt.plot(data['returns'], label=name)
    
    plt.title('Comparison of Returns Between Algorithms')
    plt.xlabel('Run')
    plt.ylabel('Return')
    plt.legend(bbox_to_anchor=(1.25, 0.5))
    plt.show()



def analyze_convergence(**kwargs):
    """Analyse and compare how quickly each algorithm converges to an optimal or near-optimal solution."""
    convergence_analysis = []

    # Iterate through each algorithm's data and get fitness and generations to best fitness
    for name, data in kwargs.items():
        avg_fitness = data['returns'].mean()
        avg_generations_to_best = data['generations_to_best_fitness'].mean()
        
        convergence_analysis.append({
            'Algorithm': name,
            'Average Fitness': avg_fitness,
            'Avg Generations to Best Fitness': avg_generations_to_best
        })
    
    convergence_comparison_table = pd.DataFrame(convergence_analysis)
    print('Convergence comparison table:')
    display(convergence_comparison_table)

    plt.figure(figsize=(12, 6))
    for i, row in convergence_comparison_table.iterrows():
        plt.scatter(row['Avg Generations to Best Fitness'], row['Average Fitness'], label=row['Algorithm'], s=100)
        plt.text(row['Avg Generations to Best Fitness'] + 0.25, row['Average Fitness'], row['Algorithm'], fontsize=8)
    
    plt.xlabel('Avg Generations to Best Fitness')
    plt.ylabel('Average Fitness')
    plt.title('Convergence analysis: Avg Fitness vs Avg Generations to Best Fitness')
    plt.legend(bbox_to_anchor=(1.25, 0.5))
    plt.show()