import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

def compare_returns(**kwargs):
    result_summary = []
    
    for name, data in kwargs.items():
        mean_return = data['returns'].mean()
        std_return = data['returns'].std()
        result_summary.append({
            'Algorithm': name,
            'Mean Return': mean_return,
            'Stability (Std Return)': std_return
        })
    
    result_df = pd.DataFrame(result_summary)
    display(result_df)
    
    # Plot comparison of returns
    plt.figure(figsize=(12, 6))
    for name, data in kwargs.items():
        plt.plot(data['returns'], label=name)
    
    plt.title('Comparison of Returns Between Algorithms')
    plt.xlabel('Run')
    plt.ylabel('Return')
    plt.legend(bbox_to_anchor=(1.25, 0.5))
    plt.show()



def plot_convergence(**kwargs):
    plt.figure(figsize=(12, 6))
    
    for name, logger in kwargs.items():
        generations = range(len(logger.generations))  # This gives you the number of generations
        best_fitness = logger.fitness  # Collects the fitness for plotting
        
        # Plot the best fitness scores over generations
        plt.plot(generations, best_fitness, label=f'Best Fitness ({name})')
    
    plt.title('Comparison of Convergence Between Algorithms')
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness')
    plt.legend(bbox_to_anchor=(1.25, 0.5))
    plt.show()