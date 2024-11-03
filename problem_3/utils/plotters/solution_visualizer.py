from typing import List, Tuple

from matplotlib import pyplot as plt
import pandas as pd

def print_solutions(vehicle_solutions: List[List[int]], best_distances: List[float], time_window_violations: List[int], max_customers_visited: int, best_solution_index):
    total_distance = sum(best_distances)
    total_violations = sum(time_window_violations)

    print(f"Best solution (Index: {best_solution_index + 1})")

    for i, solution in enumerate(vehicle_solutions):
        route_str = ' -> '.join(map(str, solution[1:-1]))  # Exclude the depot
        print(f"Route {i + 1}: {route_str} | Distance travelled: {best_distances[i]:.2f} | Time window violations: {time_window_violations[i]}")

    print(f"\nTotal number of customers visited: {max_customers_visited}/100")
    print(f"Total distance: {total_distance}")
    print(f"Total time window violations: {total_violations}")

def plot_vehicle_routes(df: pd.DataFrame, vehicle_solutions: List[List[int]]):
    depot = df.iloc[0]  # Assuming the depot is the first row (index 0)
    customer_positions = df[['Lng', 'Lat']].values

    plt.figure(figsize=(10, 8))
    plt.scatter(df['Lng'], df['Lat'], c='blue', label='Customers')
    plt.scatter(depot['Lng'], depot['Lat'], c='red', label='Depot', s=200, zorder=5)

    # Assign distinct colors to each vehicle route
    colors = plt.cm.get_cmap('tab20', len(vehicle_solutions))

    for i, solution in enumerate(vehicle_solutions):
        route_positions = customer_positions[solution]  # Get positions of the customers in this route
        plt.plot(route_positions[:, 0], route_positions[:, 1], color=colors(i), marker='o', label=f'Vehicle {i + 1}')

    plt.title("Vehicle Routes for VRPTW Solution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True)

    plt.show()

def plot_convergence(total_distances_per_iteration: List[float], n_iterations: int):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(total_distances_per_iteration) + 1), total_distances_per_iteration, marker='o')
    plt.title("ACO Convergence Plot - Total Distance per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Total Distance")

    plt.xticks(range(0, n_iterations, 5))

    plt.grid(True)
    plt.show()