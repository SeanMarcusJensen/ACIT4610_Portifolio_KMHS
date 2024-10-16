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
    
    result_df = pd.DataFrame(returns_analysis)
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



def analyze_convergence(**algorithm_data):
    """Analyse and compare how quickly each algorithm converges to an optimal or near-optimal solution."""
    # Create an empty list to collect results
    comparison_results = []

    # Iterate through each algorithm's data and collect fitness and generations to best fitness
    for name, data in algorithm_data.items():
        avg_fitness = data['returns'].mean()
        avg_generations_to_best = data['generations_to_best_fitness'].mean()
        
        # Append results to list
        comparison_results.append({
            'Algorithm': name,
            'Average Fitness': avg_fitness,
            'Avg Generations to Best Fitness': avg_generations_to_best
        })
    
    # Create a DataFrame from the results
    comparison_table = pd.DataFrame(comparison_results)
    
    # Output the table
    print("Comparison Table:")
    print(comparison_table)

    # Visualization: Fitness and Generations to Best Fitness
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # Plot Average Fitness on the left axis
    ax1.bar(comparison_table['Algorithm'], comparison_table['Average Fitness'], color='g', width=0.4, align='center')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Average Fitness', color='g')
    
    # Plot Generations to Best Fitness on the right axis
    ax2.plot(comparison_table['Algorithm'], comparison_table['Avg Generations to Best Fitness'], color='b', marker='o', linestyle='--')
    ax2.set_ylabel('Avg Generations to Best Fitness', color='b')

    plt.title('Algorithm Comparison: Fitness vs Generations to Best Fitness')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    return comparison_table