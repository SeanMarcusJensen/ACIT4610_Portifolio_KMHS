import random
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from utils.matrix.distance_matrix import DistanceMatrix

class ACOParameters:
    """Manages the parameters and pheromone levels for the ACO algorithm.

    Attributes:
        alpha (float): Weight of pheromone importance in path selection.
        beta (float): Weight of heuristic (distance) importance in path selection.
        evaporation_rate (float): Rate at which pheromone trails evaporate.
        Q (float): Constant influencing the amount of pheromone added per solution.
        pheromones (np.ndarray): A matrix storing the pheromone levels between locations.
    """
    def __init__(self, n_locations: int, alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.5, initial_pheromone: float = 0.2, Q: float = 100.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.pheromones = np.full((n_locations, n_locations), initial_pheromone)

    def update_pheromones(self, solutions: List[Tuple[List[int], float, int]]) -> None:
        self.pheromones *= (1 - self.evaporation_rate) # Pheromone evaporation

        for solution, total_cost, _ in solutions:
            # Check for invalid solution
            if total_cost == 0 or total_cost == float('inf'):
                continue

            # Update pheromone levels
            for i in range(len(solution) - 1):
                current, next = solution[i], solution[i + 1]

                # Update pheromone levels for the edges both ways
                self.pheromones[current][next] += self.Q / total_cost
                self.pheromones[next][current] += self.Q / total_cost

class Vehicle:
    """Represents a delivery vehicle for solving the VRPTW.

    Attributes:
        max_capacity (int): The maximum capacity of the vehicle.
        penalty_early (int): Penalty value for showing up at customer location early.
        penalty_late (int): Penalty value for showing up at customer location late.
    """
    def __init__(self, max_capacity: int) -> None:
        self.max_capacity = max_capacity
        self.penalty_early = 50
        self.penalty_late = 100

    def select_next_customer(self, 
                             pheromones: np.ndarray, 
                             distances: np.ndarray, 
                             current_customer: int,
                             unvisited_customers: set, 
                             alpha: float, 
                             beta: float, 
                             demands: List[int],
                             current_time: int, 
                             time_windows: List[Tuple[int, int]]) -> Optional[int]:
        probabilities = []
        total_probability = 0.0

        for next_customer in unvisited_customers:
            demand = demands[next_customer]
            ready_time, due_time = time_windows[next_customer]
            travel_time = distances[current_customer][next_customer]
            arrival_time = current_time + travel_time

            penalty = 0
            if arrival_time < ready_time:
                penalty = self.penalty_early

            if demand <= self.max_capacity:
                pheromone = pheromones[current_customer][next_customer] ** alpha # Calculate pheromone importance
                heuristic = (1 / (distances[current_customer][next_customer] + 1e-6)) ** beta # Calculate cost importance
                probability = pheromone * heuristic / (1 + penalty) # Add penalty to probability calculation if necessary
                probabilities.append((next_customer, probability))
                total_probability += probability

        if total_probability == 0:
            return None

        # Normalize probabilities
        probabilities = [(customer, probability / total_probability) for customer, probability in probabilities]
        
        # Select next customer based on cumulative probabilities (roulette wheel selection)
        rand_val = random.random()
        cumulative_probability = 0.0
        for customer, probability in probabilities:
            cumulative_probability += probability
            if rand_val <= cumulative_probability:
                return customer
        
        return None # If no customer was selected


class ACO:
    """Manages the Ant Colony Optimization process for solving the VRPTW.

    Attributes:
        n_vehicles (int): Number of vehicles (ants) available for service.
        n_iterations (int): Number of iterations to run the ACO algorithm.
        aco_params (ACOParameters): Instance of ACOParameters managing parameters and pheromone levels.
        max_capacity (int): The maximum capacity each vehicle can carry.
        locations_df (pd.DataFrame): DataFrame containing locations and parameters.
        distances (np.ndarray): Matrix of distances between each location.
        demands (List[int]): List of demands for each customer.
        time_windows (List[Tuple[int, int]]): List of time windows (ready and due times) for each customer.
        service_times (List[int]): List of time it takes to serve each customer.
    """
    def __init__(self, locations_df: pd.DataFrame, n_vehicles: int, n_iterations: int, aco_params: ACOParameters, max_capacity: int) -> None:
        self.n_vehicles = n_vehicles
        self.n_iterations = n_iterations
        self.aco_params = aco_params
        self.max_capacity = max_capacity
        self.n_locations = len(locations_df)
        self.distances = DistanceMatrix(locations_df).matrix
        self.demands = locations_df['Demand'].tolist()
        self.time_windows = list(zip(locations_df['ReadyTime'], locations_df['Due']))
        self.service_times = locations_df['ServiceTime'].tolist() 

    def construct_solution(self, vehicle: Vehicle, unvisited_customers: set) -> Tuple[List[int], float, int, int]:
        """Construct a solution route for a single vehicle"""
        current_customer = 0 # Start from the depot (Customer 0)
        solution = [current_customer] # Solution path
        current_capacity = self.max_capacity # Initialize vehicle with full capacity
        current_time = self.time_windows[0][0] # Initial time set to the ready time of depot
        total_penalty = 0
        time_window_violations = 0

        while unvisited_customers:
            # Filter feasible customers based on current capacity and due time
            feasible_customers = {
                customer for customer in unvisited_customers 
                if self.demands[customer] <= current_capacity and
                current_time + self.distances[current_customer][customer] <= self.time_windows[customer][1] # Check if arrival time is less than due time
            }

            if not feasible_customers:
                # Return to depot if no feasible customers
                break

            next_customer = vehicle.select_next_customer(
                self.aco_params.pheromones, self.distances, current_customer, feasible_customers,
                self.aco_params.alpha, self.aco_params.beta, self.demands, current_time, self.time_windows
            )

            if next_customer is None:
                # Return to depot if no next customer remains
                break

            travel_time = self.distances[current_customer][next_customer]
            arrival_time = current_time + travel_time
            ready_time, due_time = self.time_windows[next_customer]

            print(f"Visiting customer {next_customer} with current time {current_time}, travel time {travel_time}, demand {self.demands[next_customer]}")

            if arrival_time < ready_time:
                current_time = ready_time + self.service_times[next_customer]
                total_penalty += vehicle.penalty_early
                time_window_violations += 1
                current_capacity -= self.demands[next_customer]
                solution.append(next_customer)
                current_customer = next_customer
                unvisited_customers.remove(next_customer)
                print(f"Early violation at {next_customer}, current penalty {total_penalty}")
                print(f"Remaining capacity {current_capacity}, unvisited customers: {unvisited_customers}")

            else:
                current_time = arrival_time + self.service_times[next_customer]
                current_capacity -= self.demands[next_customer]
                solution.append(next_customer)
                current_customer = next_customer
                unvisited_customers.remove(next_customer)
                print(f"Arrived on time at {next_customer}")
                print(f"Remaining capacity {current_capacity}, unvisited customers: {unvisited_customers}")

            # Check if the remaining capacity can serve any unvisited customer
            if current_capacity < min([self.demands[customer] for customer in unvisited_customers], default=0):
                # Return to depot if vehicle can't serve any of the next customers
                break 

        # Return to the depot
        solution.append(0) 

        # Calculate the total cost for the constructed route
        total_distance = sum(self.distances[solution[i]][solution[i + 1]] for i in range(len(solution) - 1))
        total_cost = total_distance + total_penalty
        
        print(f"Route completed with {time_window_violations} violations")

        return solution, total_cost, current_capacity, time_window_violations

    def optimize(self) -> Tuple[List[List[int]], List[float], List[int], int, List[float], int]:
        all_vehicle_solutions = []
        all_route_distances = []
        all_route_violations = []
        all_customers_visited = []
        total_distances = [] # Total distances per iteration

        best_iteration_index = None # Index of the iteration that performed best
        max_customers_visited = 0
        min_total_distance = float('inf')
        
        for i in range(self.n_iterations):
            print(f"______________________________________________ITERATION {i + 1}______________________________________________")
            solutions = []
            vehicle_solutions = []
            route_distances = []
            route_violations = []
            customers_visited_in_iteration = 0

            unvisited_customers = set(range(1, self.n_locations)) # All customers (excluding the depot)
            
            for j in range(self.n_vehicles):
                if not unvisited_customers:
                    break
                
                print(f"VEHICLE {j + 1}")
                vehicle = Vehicle(self.max_capacity)
                solution, total_cost, remaining_capacity, violation = self.construct_solution(vehicle, unvisited_customers)
                
                if solution:
                    solutions.append((solution, total_cost, remaining_capacity))
                    vehicle_solutions.append(solution)
                    route_distances.append(total_cost)
                    route_violations.append(violation)
                    customers_visited_in_iteration += len(solution) - 2 # Exclude depot from customer count

            # Update pheromones based on solutions
            self.aco_params.update_pheromones(solutions)
            print(f"Pheromone values: {self.aco_params.pheromones}")

            # Store solutions, distances, and violations for this iteration
            all_vehicle_solutions.append(vehicle_solutions)
            all_route_distances.append(route_distances)
            all_route_violations.append(route_violations)
            all_customers_visited.append(customers_visited_in_iteration)

            total_distance_in_iteration = sum(route_distances)
            total_distances.append(total_distance_in_iteration)
            print(f"Iteration {i + 1} total distance: {total_distance_in_iteration}")

            # Check if this iteration has the best solution so far
            if (customers_visited_in_iteration > max_customers_visited or (customers_visited_in_iteration == max_customers_visited and total_distance_in_iteration < min_total_distance)):
                best_iteration_index = i
                max_customers_visited = customers_visited_in_iteration
                min_total_distance = total_distance_in_iteration

        if best_iteration_index is None:
            raise ValueError("No valid solution found across all iterations")

        # Return solutions for the best iteration
        best_vehicle_solutions = all_vehicle_solutions[best_iteration_index]
        best_distances = all_route_distances[best_iteration_index]
        time_window_violations = all_route_violations[best_iteration_index]
        
        return best_vehicle_solutions, best_distances, time_window_violations, max_customers_visited, total_distances, best_iteration_index
